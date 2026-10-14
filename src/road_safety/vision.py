"""India road-damage detection with a range-read RDD2022 data path.

The RDD2022 Figshare item is a 13.26 GB ZIP. This module uses HTTP range
requests and Python's ZIP reader to inspect it and extract only annotated
India training images, avoiding a full archive download. Model training is
optional and uses a COCO-pretrained MobileNetV3 Faster R-CNN detector.

Road defects are a separate infrastructure evidence layer. RDD2022 images are
not geolinked to the crash records used by the broader road-safety project.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, OrderedDict, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable


FIGSHARE_ARTICLE_ID = 21431547
FIGSHARE_DOI = "10.6084/m9.figshare.21431547.v1"
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"
FIGSHARE_PAGE_URL = (
    "https://figshare.com/articles/dataset/"
    "RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547"
)
CHALLENGE_DATA_URL = "https://crddc2022.sekilab.global/data/"
CHALLENGE_RULES_URL = "https://crddc2022.sekilab.global/rules/"
OFFICIAL_INDIA_ZIP_URL = (
    "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/"
    "Country_Specific_Data_CRDDC2022/RDD2022_India.zip"
)
EXPECTED_ARCHIVE_BYTES = 13_264_172_619
EXPECTED_ARCHIVE_MD5 = "b62bd51d2ffcfaa76c60f234f0cc2bb3"
CLASSES = {
    "D00": "longitudinal_crack",
    "D10": "transverse_crack",
    "D20": "alligator_crack",
    "D40": "pothole",
}
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASSES, start=1)}
ID_TO_LABEL = {idx: name for name, idx in LABEL_TO_ID.items()}
USER_AGENT = "india-road-safety-portfolio/0.1 (selective Figshare ZIP range reader)"


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_raw_root() -> Path:
    return _project_root() / "data" / "raw" / "rdd2022"


def _default_processed_root() -> Path:
    return _project_root() / "data" / "processed" / "vision"


class HTTPRangeReader(io.RawIOBase):
    """Small seekable file object backed by HTTP Range GETs and an LRU block cache."""

    def __init__(
        self,
        url: str,
        *,
        block_size: int = 8 * 1024 * 1024,
        cached_blocks: int = 8,
        timeout: int = 90,
        retries: int = 4,
    ) -> None:
        super().__init__()
        self.url = url
        self.block_size = block_size
        self.cached_blocks = cached_blocks
        self.timeout = timeout
        self.retries = retries
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.requests = 0
        self.bytes_received = 0
        self.size = self._probe_size()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError(f"unsupported seek mode: {whence}")
        if target < 0:
            raise ValueError("negative seek position")
        self.position = target
        return target

    def _range_request(self, start: int, end: int) -> tuple[bytes, str | None]:
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT,
                     "Accept-Encoding": "identity"},
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    if response.status != 206:
                        raise RuntimeError(
                            f"server returned HTTP {response.status} instead of range status 206; "
                            "refusing a full-file fallback"
                        )
                    content_range = response.headers.get("Content-Range")
                    expected_prefix = f"bytes {start}-{end}/"
                    if not content_range or not content_range.startswith(expected_prefix):
                        raise RuntimeError(
                            f"unexpected Content-Range {content_range!r}; wanted {expected_prefix}"
                        )
                    data = response.read(end - start + 1)
                    if len(data) != end - start + 1:
                        raise IOError(
                            f"short range response at {start}: received {len(data)} bytes"
                        )
                    return data, content_range
            except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
                last_error = exc
                if isinstance(exc, RuntimeError) and "instead of range status" in str(exc):
                    raise
                if attempt + 1 < self.retries:
                    time.sleep(min(2**attempt, 8))
        assert last_error is not None
        raise last_error

    def _probe_size(self) -> int:
        data, content_range = self._range_request(0, 0)
        if len(data) != 1 or not content_range:
            raise RuntimeError("could not establish the remote archive size from a byte range")
        try:
            total = int(content_range.rsplit("/", 1)[1])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"invalid Content-Range header: {content_range}") from exc
        if total <= 0:
            raise RuntimeError(f"invalid remote archive size: {total}")
        self.requests += 1
        self.bytes_received += 1
        return total

    def _block(self, block_start: int) -> bytes:
        if block_start not in self.cache:
            block_end = min(block_start + self.block_size, self.size) - 1
            data, _ = self._range_request(block_start, block_end)
            self.cache[block_start] = data
            self.requests += 1
            self.bytes_received += len(data)
            self.cache.move_to_end(block_start)
            while len(self.cache) > self.cached_blocks:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(block_start)
        return self.cache[block_start]

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed range reader")
        if size is None or size < 0:
            size = self.size - self.position
            if size > 32 * 1024 * 1024:
                raise RuntimeError(
                    "unbounded read exceeded the 32 MiB safety limit; ZIP range access aborted"
                )
        if size == 0 or self.position >= self.size:
            return b""
        remaining = min(size, self.size - self.position)
        pieces: list[bytes] = []
        while remaining:
            block_start = (self.position // self.block_size) * self.block_size
            block = self._block(block_start)
            in_block = self.position - block_start
            take = min(remaining, len(block) - in_block)
            if take <= 0:
                break
            pieces.append(block[in_block:in_block + take])
            self.position += take
            remaining -= take
        return b"".join(pieces)


def _figshare_file_metadata() -> dict[str, Any]:
    request = urllib.request.Request(FIGSHARE_API_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        article = json.loads(response.read().decode("utf-8"))
    files = article.get("files", [])
    matches = [
        item for item in files
        if item.get("size") == EXPECTED_ARCHIVE_BYTES
        or item.get("supplied_md5") == EXPECTED_ARCHIVE_MD5
    ]
    if not matches:
        raise RuntimeError("Figshare API response did not include the expected RDD2022 ZIP")
    chosen = matches[0]
    if chosen.get("size") != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError(f"Figshare archive size changed: {chosen.get('size')}")
    return {"article": article, "file": chosen}


def _is_india_training_image_member(name: str) -> bool:
    normalized = name.replace("\\", "/").lstrip("./")
    parts = [part.casefold() for part in PurePosixPath(normalized).parts]
    return "india" in parts and "train" in parts and normalized.casefold().endswith(".jpg")


def _is_india_training_annotation_member(name: str) -> bool:
    normalized = name.replace("\\", "/").lstrip("./")
    parts = [part.casefold() for part in PurePosixPath(normalized).parts]
    return "india" in parts and "train" in parts and normalized.casefold().endswith(".xml")


def remote_archive_index(processed_root: Path) -> dict[str, Any]:
    metadata = _figshare_file_metadata()
    file_info = metadata["file"]
    url = file_info["download_url"]
    reader = HTTPRangeReader(url)
    try:
        with zipfile.ZipFile(reader, "r") as archive:
            outer_members = archive.infolist()
            india_entry = next(
                (entry for entry in outer_members
                 if PurePosixPath(entry.filename.replace("\\", "/")).name.casefold() == "india.zip"),
                None,
            )
            if india_entry is None:
                raise RuntimeError("The Figshare ZIP no longer contains its nested India.zip member")
            with archive.open(india_entry, "r") as india_stream:
                with zipfile.ZipFile(india_stream, "r") as india_archive:
                    members = india_archive.infolist()
                    selected = [
                        entry for entry in members
                        if not entry.is_dir() and (
                            _is_india_training_image_member(entry.filename)
                            or _is_india_training_annotation_member(entry.filename)
                        )
                    ]
                    images = [entry for entry in selected if _is_india_training_image_member(entry.filename)]
                    annotations = [
                        entry for entry in selected if _is_india_training_annotation_member(entry.filename)
                    ]
                    other_entries = [
                        entry for entry in members
                        if not entry.is_dir() and entry not in selected
                    ]
                    offsets = sorted((entry.header_offset, entry.filename, entry.file_size) for entry in selected)
                    offsets_other = [entry.header_offset for entry in other_entries]
                    span_start = offsets[0][0] if offsets else None
                    span_end = max((entry.header_offset + entry.compress_size for entry in selected), default=None)
                    interleaved_non_selected = (
                        sum(1 for offset in offsets_other if span_start is not None and span_end is not None
                            and span_start <= offset <= span_end)
                    )
            stats = {
                "dataset": "RDD2022 / India / train",
                "figshare_doi": FIGSHARE_DOI,
                "archive_name": file_info.get("name"),
                "archive_size_bytes": file_info.get("size"),
                "archive_md5": file_info.get("computed_md5") or file_info.get("supplied_md5"),
                "archive_total_members": len(outer_members),
                "outer_archive_india_member": india_entry.filename,
                "outer_archive_india_member_bytes": india_entry.file_size,
                "india_archive_total_members": len(members),
                "india_train_image_members": len(images),
                "india_train_annotation_members": len(annotations),
                "india_train_compressed_bytes": sum(entry.compress_size for entry in selected),
                "india_train_uncompressed_bytes": sum(entry.file_size for entry in selected),
                "india_train_first_member_offset": span_start,
                "india_train_last_member_end_offset": span_end,
                "non_train_members_interleaved_in_india_train_span": interleaved_non_selected,
                "member_path_examples": [entry.filename for entry in selected[:5]],
                "selection": "Only India/train .jpg and .xml files; no test or other-country members.",
                "range_reader_requests_for_index": reader.requests,
                "range_reader_bytes_for_index": reader.bytes_received,
                "range_reader_block_size_bytes": reader.block_size,
                "indexed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
    finally:
        reader.close()
    out = processed_root / "remote_archive_profile.json"
    _json_write(out, stats)
    return stats


def _safe_member_path(root: Path, member_name: str) -> Path:
    target = (root / Path(*PurePosixPath(member_name.replace("\\", "/")).parts)).resolve()
    resolved_root = root.resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"unsafe ZIP member path rejected: {member_name}") from exc
    return target


def _manifest_payload(archive_profile: dict[str, Any], *, extracted: bool) -> dict[str, Any]:
    return {
        "dataset": "RDD2022 India training split",
        "prepared_date": date.today().isoformat(),
        "release": {
            "title": "RDD2022 - The multi-national Road Damage Dataset released through CRDDC 2022",
            "doi": FIGSHARE_DOI,
            "figshare_article": FIGSHARE_PAGE_URL,
            "figshare_api": FIGSHARE_API_URL,
            "figshare_file_name": archive_profile.get("archive_name"),
            "figshare_file_size_bytes": archive_profile.get("archive_size_bytes"),
            "figshare_file_md5": archive_profile.get("archive_md5"),
            "figshare_file_md5_provenance": "Figshare API computed_md5 metadata; not locally recomputed",
            "challenge_page_india_train_image_count_approx": 9000,
            "observed_india_train_image_count_in_figshare_archive": (
                archive_profile.get("india_train_image_members")
            ),
            "image_count_note": (
                "The CRDDC page rounds India train images to about 9,000; the Figshare India.zip "
                "member contains the exact archive-enumerated count recorded here."
            ),
            "challenge_source": CHALLENGE_DATA_URL,
            "challenge_rules_source": CHALLENGE_RULES_URL,
            "official_country_zip_url": OFFICIAL_INDIA_ZIP_URL,
            "official_country_zip_observed_http_status": 403,
            "retrieval_method": (
                "Figshare ZIP central-directory inspection and selective India/train member reads "
                "using HTTP byte ranges; no full-archive save"
            ),
            "license_figshare_metadata": "CC BY 4.0",
            "license_challenge_page": "CC BY-SA 4.0 (challenge image license statement)",
            "license_handling": (
                "The repository metadata and challenge page state different CC licenses. Preserve "
                "source attribution and this conflict; confirm the relevant file/image terms before "
                "redistribution. This portfolio does not republish the source images."
            ),
            "retrieved_date": date.today().isoformat() if extracted else None,
        },
        "geography_and_period": {
            "country": "India",
            "collection_description": "Delhi and Gurugram metropolitan areas, with other collection areas mainly in Haryana",
            "collection_vintage": "RDD2022 release; collection period is not a crash-event period",
            "geolocation_available": False,
        },
        "labels": CLASSES,
        "use_limitations": [
            "Images are road-surface condition examples and are not linked to any crash record, date, or coordinate.",
            "A detected defect cannot be described as a cause or predictor of an observed crash.",
            "The challenge test split has no public annotations and is not used for this evaluation.",
            "Collection areas and conditions do not represent every Indian road or season.",
        ],
        "selective_archive_profile": archive_profile,
    }


def extract_india_training(raw_root: Path, processed_root: Path, max_compressed_gb: float) -> dict[str, Any]:
    profile = remote_archive_index(processed_root)
    expected_train_members = (
        profile["india_train_image_members"] + profile["india_train_annotation_members"]
    )
    if not profile["india_train_image_members"] or not profile["india_train_annotation_members"]:
        raise RuntimeError("No India/train image and annotation members found; extraction stopped")
    compressed_bytes = profile["india_train_compressed_bytes"]
    limit_bytes = int(max_compressed_gb * 1024**3)
    if compressed_bytes > limit_bytes:
        raise RuntimeError(
            f"India/train compressed members total {compressed_bytes:,} bytes, above the "
            f"configured {limit_bytes:,}-byte download cap"
        )

    source_meta = _figshare_file_metadata()["file"]
    reader = HTTPRangeReader(source_meta["download_url"])
    root = raw_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    extracted_count = 0
    extracted_bytes = 0
    started = time.time()
    try:
        with zipfile.ZipFile(reader, "r") as archive:
            india_entry = next(
                (entry for entry in archive.infolist()
                 if PurePosixPath(entry.filename.replace("\\", "/")).name.casefold() == "india.zip"),
                None,
            )
            if india_entry is None:
                raise RuntimeError("India.zip member disappeared from the Figshare archive")
            with archive.open(india_entry, "r") as india_stream:
                with zipfile.ZipFile(india_stream, "r") as india_archive:
                    selected = [
                        entry for entry in india_archive.infolist()
                        if not entry.is_dir() and (
                            _is_india_training_image_member(entry.filename)
                            or _is_india_training_annotation_member(entry.filename)
                        )
                    ]
                    selected.sort(key=lambda entry: entry.header_offset)
                    for index, entry in enumerate(selected, start=1):
                        target = _safe_member_path(root, entry.filename)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with india_archive.open(entry, "r") as source, target.open("wb") as destination:
                            shutil.copyfileobj(source, destination, length=1024 * 1024)
                        extracted_count += 1
                        extracted_bytes += target.stat().st_size
                        if index % 1000 == 0 or index == len(selected):
                            elapsed = max(time.time() - started, 0.001)
                            print(
                                f"extracted {index:,}/{len(selected):,} India/train files; "
                                f"{extracted_bytes / 1024**2:,.1f} MiB written; "
                                f"{reader.bytes_received / 1024**2:,.1f} MiB range data read; "
                                f"{extracted_bytes / 1024**2 / elapsed:,.1f} MiB/s output",
                                flush=True,
                            )
    finally:
        range_stats = {
            "range_reader_requests": reader.requests,
            "range_reader_bytes": reader.bytes_received,
            "range_reader_block_size_bytes": reader.block_size,
        }
        reader.close()

    manifest = _manifest_payload(profile, extracted=True)
    manifest["extraction"] = {
        "selected_archive_members": expected_train_members,
        "extracted_files": extracted_count,
        "extracted_bytes": extracted_bytes,
        "transport_range_requests": range_stats["range_reader_requests"],
        "transport_bytes_received_including_cache_blocks": range_stats["range_reader_bytes"],
        "elapsed_seconds": round(time.time() - started, 2),
        "whole_archive_downloaded": False,
        "full_archive_md5_locally_verified": False,
        "figshare_full_archive_md5_is_upstream_metadata_only": True,
        "selected_inner_train_member_crc_checked_by_zipfile": True,
        "outer_india_zip_crc_checked": False,
    }
    _json_write(raw_root / "source_manifest.json", manifest)
    if extracted_count != expected_train_members:
        raise RuntimeError(
            f"member count mismatch: expected {expected_train_members}, extracted {extracted_count}"
        )
    return manifest["extraction"]


def _india_train_directories(raw_root: Path) -> list[Path]:
    candidates = [
        path for path in raw_root.rglob("*")
        if path.is_dir() and path.name.casefold() == "train"
        and any(part.casefold() == "india" for part in path.parts)
    ]
    return sorted(candidates)


def _find_image(xml_path: Path, xml_root: ET.Element, train_dir: Path,
                image_lookup: dict[str, Path]) -> Path | None:
    filename = (xml_root.findtext("filename") or "").strip()
    candidates = []
    if filename:
        indexed = image_lookup.get(Path(filename).name.casefold())
        if indexed is not None:
            return indexed
        candidates.append(train_dir / Path(filename).name)
        candidates.append(xml_path.with_name(Path(filename).name))
        candidates.append(xml_path.with_suffix(".jpg"))
        candidates.append(xml_path.with_suffix(".JPG"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    stem = Path(filename).stem if filename else xml_path.stem
    for candidate in train_dir.rglob(f"{stem}.*"):
        if candidate.suffix.casefold() in {".jpg", ".jpeg"}:
            return candidate
    return None


def _read_voc(xml_path: Path, train_dir: Path,
              image_lookup: dict[str, Path]) -> tuple[
                  Path | None, list[dict[str, Any]], list[str], list[str]
              ]:
    issues: list[str] = []
    out_of_scope_classes: set[str] = set()
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as exc:
        return None, [], [f"xml_parse_error:{type(exc).__name__}"], []
    image_path = _find_image(xml_path, root, train_dir, image_lookup)
    boxes: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        raw_class = (obj.findtext("name") or "").strip()
        if raw_class not in CLASSES:
            issues.append(f"unknown_class:{raw_class or '<empty>'}")
            out_of_scope_classes.add(raw_class or "<empty>")
            continue
        box = obj.find("bndbox")
        if box is None:
            issues.append("missing_bndbox")
            continue
        try:
            coords = [float(box.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax")]
        except ValueError:
            issues.append("invalid_box_coordinate")
            continue
        if not all(math.isfinite(value) for value in coords) or coords[2] <= coords[0] or coords[3] <= coords[1]:
            issues.append("nonpositive_or_nonfinite_box")
            continue
        boxes.append({"class_code": raw_class, "class_name": CLASSES[raw_class], "xyxy": coords})
    if image_path is None:
        issues.append("missing_image")
    return image_path, boxes, issues, sorted(out_of_scope_classes)


def profile_local_data(raw_root: Path, processed_root: Path) -> dict[str, Any]:
    train_dirs = _india_train_directories(raw_root)
    if not train_dirs:
        raise FileNotFoundError(
            f"No India/train directory under {raw_root}. Run `download-india` first."
        )
    all_records: list[dict[str, Any]] = []
    parse_issue_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    out_of_scope_class_counts: Counter[str] = Counter()
    image_seen: set[str] = set()
    for train_dir in train_dirs:
        image_lookup = {
            path.name.casefold(): path
            for path in train_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg"}
        }
        for xml_path in sorted(train_dir.rglob("*.xml")):
            image_path, boxes, issues, out_of_scope_classes = _read_voc(
                xml_path, train_dir, image_lookup
            )
            for issue in issues:
                parse_issue_counts[issue] += 1
            if image_path is None:
                continue
            image_key = image_path.resolve().as_posix()
            if image_key in image_seen:
                parse_issue_counts["duplicate_image_annotation_path"] += 1
                continue
            image_seen.add(image_key)
            for box in boxes:
                class_counts[box["class_code"]] += 1
            for issue in issues:
                if issue.startswith("unknown_class:"):
                    out_of_scope_class_counts[issue.split(":", 1)[1]] += 1
            all_records.append({
                "image_id": image_path.stem,
                "image_path": str(image_path.resolve()),
                "annotation_path": str(xml_path.resolve()),
                "object_count": len(boxes),
                "boxes": boxes,
                "out_of_scope_class_codes": out_of_scope_classes,
                "issues": issues,
            })
    all_records.sort(key=lambda record: (record["image_id"].casefold(), record["image_path"]))
    summary = {
        "dataset": "RDD2022 India training split",
        "profiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_directories": [str(path.resolve()) for path in train_dirs],
        "annotated_images_with_matching_file": len(all_records),
        "annotations_by_damage_class": dict(class_counts),
        "out_of_scope_annotation_instances_by_class": dict(out_of_scope_class_counts),
        "images_with_at_least_one_valid_box": sum(record["object_count"] > 0 for record in all_records),
        "images_with_no_valid_box": sum(record["object_count"] == 0 for record in all_records),
        "images_excluded_due_to_out_of_scope_labels": sum(
            bool(record["out_of_scope_class_codes"]) for record in all_records
        ),
        "images_eligible_for_four_class_training_and_evaluation": sum(
            not record["out_of_scope_class_codes"] for record in all_records
        ),
        "eligible_empty_background_images": sum(
            record["object_count"] == 0 and not record["out_of_scope_class_codes"]
            for record in all_records
        ),
        "issue_counts": dict(parse_issue_counts),
        "training_eligible_images": sum(not record["out_of_scope_class_codes"] for record in all_records),
        "limitation": (
            "RDD2022 images have no verified geocoordinates and are not joined to crash rows. "
            "Images bearing annotations outside the four target classes are excluded from fitting and "
            "evaluation so those documented defects are not treated as background. Empty XML labels are "
            "retained as negatives only when no out-of-scope label is present. The available country-level "
            "subset does not identify location or road segment per image."
        ),
    }
    _json_write(processed_root / "dataset_profile.json", summary)
    _json_write(processed_root / "voc_records.json", all_records)
    return summary


def _stable_group_key(image_id: str, ordinal: int, group_size: int) -> str:
    suffix = re.search(r"(\d+)$", image_id)
    if suffix:
        bucket = int(suffix.group(1)) // group_size
        return f"numeric_id_block_{bucket:06d}"
    return f"filename_order_block_{ordinal // group_size:06d}"


def create_grouped_split(
    records: list[dict[str, Any]],
    *,
    seed: int = 42,
    validation_fraction: float = 0.2,
    group_size: int = 100,
) -> dict[str, Any]:
    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be between 0.05 and 0.5")
    if group_size < 1:
        raise ValueError("group_size must be at least one")
    ordered = sorted(records, key=lambda row: (row["image_id"].casefold(), row["image_path"]))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ordinal, record in enumerate(ordered):
        group = _stable_group_key(record["image_id"], ordinal, group_size)
        groups[group].append(record)
    group_names = sorted(groups)
    ranked = sorted(
        group_names,
        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest(),
    )
    validation_group_count = max(1, round(len(group_names) * validation_fraction))
    if len(group_names) > 1:
        validation_group_count = min(validation_group_count, len(group_names) - 1)
    validation_groups = set(ranked[:validation_group_count])
    train_ids = [row["image_id"] for name, rows in groups.items() if name not in validation_groups for row in rows]
    validation_ids = [row["image_id"] for name, rows in groups.items() if name in validation_groups for row in rows]
    return {
        "seed": seed,
        "validation_fraction_requested": validation_fraction,
        "group_size_numeric_id_or_sorted_filename_block": group_size,
        "group_assignment": "SHA-256 ranked deterministic holdout by filename-derived block",
        "is_geographic_holdout": False,
        "grouping_caveat": (
            "RDD2022 filenames and country labels do not provide verified coordinates or capture routes. "
            "Blocks of sequential numeric filename IDs are only a sequence proxy; they are not confirmed "
            "geographic groups. This prevents records in one filename block crossing splits, but cannot "
            "prove independence between nearby frames or locations."
        ),
        "train_group_count": len(group_names) - len(validation_groups),
        "validation_group_count": len(validation_groups),
        "train_image_ids": sorted(train_ids),
        "validation_image_ids": sorted(validation_ids),
    }


def build_split_file(raw_root: Path, processed_root: Path, seed: int, validation_fraction: float,
                     group_size: int) -> dict[str, Any]:
    rows_path = processed_root / "voc_records.json"
    profile_path = processed_root / "dataset_profile.json"
    if not rows_path.exists() or not profile_path.exists():
        profile = profile_local_data(raw_root, processed_root)
        rows = json.loads(rows_path.read_text(encoding="utf-8"))
    else:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        rows = json.loads(rows_path.read_text(encoding="utf-8"))
    eligible_rows = [row for row in rows if not row.get("out_of_scope_class_codes")]
    split = create_grouped_split(
        eligible_rows, seed=seed, validation_fraction=validation_fraction, group_size=group_size
    )
    split["images_excluded_for_out_of_scope_labels"] = len(rows) - len(eligible_rows)
    split["dataset_profile"] = profile
    _json_write(processed_root / "grouped_split.json", split)
    return split


def _torch_components() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except ImportError as exc:
        raise SystemExit(
            "Vision dependencies are missing. From the project directory run: "
            "uv sync --extra vision"
        ) from exc
    return torch, Image, DataLoader, Dataset, (torchvision, FastRCNNPredictor)


def _load_records(processed_root: Path) -> list[dict[str, Any]]:
    path = processed_root / "voc_records.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `profile` before training")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        row for row in rows
        if Path(row["image_path"]).exists() and not row.get("out_of_scope_class_codes")
    ]


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cap_ids(ids: set[str], records: list[dict[str, Any]], maximum: int | None,
             seed: int) -> set[str]:
    """Deterministically cap IDs while reserving at least one sample per available class."""
    if maximum is None or len(ids) <= maximum:
        return set(ids)
    rows = {record["image_id"]: record for record in records if record["image_id"] in ids}
    order_key = lambda item: hashlib.sha256(f"{seed}:{item}".encode("utf-8")).hexdigest()
    by_class: dict[str, list[str]] = defaultdict(list)
    for image_id, record in rows.items():
        for code in {box["class_code"] for box in record["boxes"]}:
            by_class[code].append(image_id)
    selected: set[str] = set()
    for code, candidates in sorted(by_class.items(), key=lambda item: (len(item[1]), item[0])):
        if any(
            box["class_code"] == code
            for image_id in selected
            for box in rows[image_id]["boxes"]
        ):
            continue
        if len(selected) >= maximum:
            break
        selected.add(min(candidates, key=order_key))
    remaining = sorted(ids - selected, key=order_key)
    selected.update(remaining[:max(0, maximum - len(selected))])
    return selected


def _to_dataset_class(torch: Any, Image: Any, Dataset: Any, records: list[dict[str, Any]],
                      selected_ids: set[str]):
    class VOCIndiaDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = [row for row in rows if row["image_id"] in selected_ids]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            record = self.rows[index]
            with Image.open(record["image_path"]) as image:
                image = image.convert("RGB")
                width, height = image.size
                image_tensor = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
            boxes: list[list[float]] = []
            labels: list[int] = []
            for item in record["boxes"]:
                x1, y1, x2, y2 = item["xyxy"]
                x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
                y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(LABEL_TO_ID[item["class_code"]])
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([index], dtype=torch.int64),
                "area": torch.as_tensor(
                    [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes], dtype=torch.float32
                ),
                "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            }
            return image_tensor, target, record

    return VOCIndiaDataset


def _make_model(torchvision: Any, FastRCNNPredictor: Any, pretrained: bool):
    weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    if pretrained:
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    else:
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None,
                                                                                    weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES) + 1)
    return model


def _collate_detection(batch: list[Any]):
    images, targets, records = zip(*batch)
    return list(images), list(targets), list(records)


def _box_iou_xyxy(box: Any, boxes: Any) -> Any:
    import numpy as np
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    area_a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def _average_precision(recalls: Any, precisions: Any) -> float:
    import numpy as np
    recall = np.concatenate(([0.0], np.asarray(recalls), [1.0]))
    precision = np.concatenate(([0.0], np.asarray(precisions), [0.0]))
    for index in range(len(precision) - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    change = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[change + 1] - recall[change]) * precision[change + 1]))


def _evaluate_detections(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    *,
    confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    import numpy as np
    thresholds = [round(value, 2) for value in np.arange(0.50, 0.951, 0.05)]
    per_class: dict[str, Any] = {}
    all_tp = all_fp = all_fn = 0
    for label_id, class_name in ID_TO_LABEL.items():
        class_truth: dict[int, Any] = {}
        class_gt_count = 0
        for image_index, target in enumerate(ground_truth):
            mask = np.asarray(target["labels"]) == label_id
            boxes = np.asarray(target["boxes"], dtype=np.float32).reshape(-1, 4)[mask]
            class_truth[image_index] = {"boxes": boxes, "matched": np.zeros(len(boxes), dtype=bool)}
            class_gt_count += len(boxes)
        class_predictions: list[tuple[float, int, Any]] = []
        for image_index, prediction in enumerate(predictions):
            boxes = np.asarray(prediction["boxes"], dtype=np.float32).reshape(-1, 4)
            labels = np.asarray(prediction["labels"], dtype=np.int64)
            scores = np.asarray(prediction["scores"], dtype=np.float32)
            for box, label, score in zip(boxes, labels, scores):
                if label == label_id:
                    class_predictions.append((float(score), image_index, box))
        class_predictions.sort(key=lambda row: row[0], reverse=True)
        ap_by_threshold: dict[str, float | None] = {}
        pr50: tuple[int, int, int] = (0, 0, class_gt_count)
        for threshold in thresholds:
            for row in class_truth.values():
                row["matched"][:] = False
            tp_flags: list[float] = []
            fp_flags: list[float] = []
            for score, image_index, predicted_box in class_predictions:
                truth = class_truth[image_index]
                overlaps = _box_iou_xyxy(predicted_box, truth["boxes"])
                best = int(np.argmax(overlaps)) if len(overlaps) else -1
                if best >= 0 and overlaps[best] >= threshold and not truth["matched"][best]:
                    truth["matched"][best] = True
                    tp_flags.append(1.0)
                    fp_flags.append(0.0)
                else:
                    tp_flags.append(0.0)
                    fp_flags.append(1.0)
            if class_gt_count:
                tp_curve = np.cumsum(tp_flags)
                fp_curve = np.cumsum(fp_flags)
                recalls = tp_curve / class_gt_count
                precisions = tp_curve / np.maximum(tp_curve + fp_curve, 1e-12)
                ap_by_threshold[f"{threshold:.2f}"] = _average_precision(recalls, precisions)
            else:
                ap_by_threshold[f"{threshold:.2f}"] = None
            if math.isclose(threshold, 0.50):
                conf_tp = conf_fp = 0
                for row in class_truth.values():
                    row["matched"][:] = False
                for score, image_index, predicted_box in class_predictions:
                    if score < confidence_threshold:
                        continue
                    truth = class_truth[image_index]
                    overlaps = _box_iou_xyxy(predicted_box, truth["boxes"])
                    best = int(np.argmax(overlaps)) if len(overlaps) else -1
                    if best >= 0 and overlaps[best] >= threshold and not truth["matched"][best]:
                        truth["matched"][best] = True
                        conf_tp += 1
                    else:
                        conf_fp += 1
                conf_fn = class_gt_count - conf_tp
                pr50 = (conf_tp, conf_fp, conf_fn)
        tp, fp, fn = pr50
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        valid_ap = [value for value in ap_by_threshold.values() if value is not None]
        per_class[class_name] = {
            "ground_truth_instances": class_gt_count,
            "predicted_instances": len(class_predictions),
            "precision_at_iou50_confidence_threshold": precision,
            "recall_at_iou50_confidence_threshold": recall,
            "f1_at_iou50_confidence_threshold": f1,
            "ap50": ap_by_threshold.get("0.50"),
            "ap50_95": (sum(valid_ap) / len(valid_ap)) if valid_ap else None,
            "ap_by_iou_threshold": ap_by_threshold,
        }
        all_tp += tp
        all_fp += fp
        all_fn += fn
    micro_precision = all_tp / (all_tp + all_fp) if all_tp + all_fp else 0.0
    micro_recall = all_tp / (all_tp + all_fn) if all_tp + all_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall else 0.0
    )
    aps50 = [item["ap50"] for item in per_class.values() if item["ap50"] is not None]
    aps5095 = [item["ap50_95"] for item in per_class.values() if item["ap50_95"] is not None]
    return {
        "iou_thresholds_for_ap": thresholds,
        "confidence_threshold_for_precision_recall_f1": confidence_threshold,
        "micro_precision_at_iou50": micro_precision,
        "micro_recall_at_iou50": micro_recall,
        "micro_f1_at_iou50": micro_f1,
        "map50_macro_over_classes_with_ground_truth": sum(aps50) / len(aps50) if aps50 else None,
        "map50_95_macro_over_classes_with_ground_truth": sum(aps5095) / len(aps5095) if aps5095 else None,
        "per_class": per_class,
    }


def train_detector(
    raw_root: Path,
    processed_root: Path,
    output_root: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_train_images: int | None,
    max_val_images: int | None,
    confidence_threshold: float,
    no_pretrained: bool,
) -> dict[str, Any]:
    torch, Image, DataLoader, Dataset, model_components = _torch_components()
    torchvision, FastRCNNPredictor = model_components
    records = _load_records(processed_root)
    split_path = processed_root / "grouped_split.json"
    if not split_path.exists():
        build_split_file(raw_root, processed_root, seed, 0.2, 100)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = set(split["train_image_ids"])
    val_ids = set(split["validation_image_ids"])
    eligible = {record["image_id"] for record in records}
    train_ids &= eligible
    val_ids &= eligible
    positive_train_ids = {
        record["image_id"] for record in records if record["object_count"] > 0
    }
    eligible_train_ids = set(train_ids)
    train_ids &= positive_train_ids
    train_background_images_excluded = len(eligible_train_ids - train_ids)
    available_train_images = len(train_ids)
    available_validation_images = len(val_ids)
    train_ids = _cap_ids(train_ids, records, max_train_images, seed)
    val_ids = _cap_ids(val_ids, records, max_val_images, seed + 1)
    if not train_ids or not val_ids:
        raise ValueError("training and validation splits both need at least one image")

    _seed_everything(torch, seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    train_dataset_class = _to_dataset_class(torch, Image, Dataset, records, train_ids)
    val_dataset_class = _to_dataset_class(torch, Image, Dataset, records, val_ids)
    train_ds = train_dataset_class(records)
    val_ds = val_dataset_class(records)
    record_by_id = {record["image_id"]: record for record in records}
    selected_training_class_counts = Counter(
        box["class_code"]
        for image_id in train_ids for box in record_by_id[image_id]["boxes"]
    )
    selected_validation_class_counts = Counter(
        box["class_code"]
        for image_id in val_ids for box in record_by_id[image_id]["boxes"]
    )
    selected_split = {
        "seed": seed,
        "train_image_count": len(train_ids),
        "validation_image_count": len(val_ids),
        "train_class_instance_counts": dict(selected_training_class_counts),
        "validation_class_instance_counts": dict(selected_validation_class_counts),
        "train_image_ids": sorted(train_ids),
        "validation_image_ids": sorted(val_ids),
        "selection_policy": (
            "Stable SHA-256 sample cap with one selected image for every target class present "
            "before filling remaining slots; training excludes clean empty-target images."
        ),
    }
    _json_write(processed_root / "selected_run_split.json", selected_split)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
                              collate_fn=_collate_detection)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=_collate_detection)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(torchvision, FastRCNNPredictor, pretrained=not no_pretrained).to(device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=learning_rate,
        weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 2), gamma=0.5)
    history: list[dict[str, Any]] = []
    best_map = -1.0
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    output_root.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        started = time.time()
        for step, (images, targets, _) in enumerate(train_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            loss_map = model(images, targets)
            loss = sum(loss_map.values())
            if not torch.isfinite(loss):
                batch_ids = [record["image_id"] for record in _]
                component_values = {key: float(value.detach().cpu()) for key, value in loss_map.items()}
                box_counts = [int(target["boxes"].shape[0]) for target in targets]
                raise RuntimeError(
                    f"non-finite loss in epoch {epoch}, batch {step}: {loss.item()}; "
                    f"images={batch_ids}; box_counts={box_counts}; components={component_values}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            if step % 50 == 0 or step == len(train_loader):
                print(f"epoch {epoch}/{epochs} batch {step}/{len(train_loader)} loss={loss.item():.4f}",
                      flush=True)
        scheduler.step()
        model.eval()
        predictions: list[dict[str, Any]] = []
        ground_truth: list[dict[str, Any]] = []
        with torch.inference_mode():
            for images, targets, _ in val_loader:
                outputs = model([image.to(device) for image in images])
                for prediction, target in zip(outputs, targets):
                    predictions.append({
                        key: value.detach().cpu().numpy()
                        for key, value in prediction.items()
                        if key in {"boxes", "labels", "scores"}
                    })
                    ground_truth.append({
                        "boxes": target["boxes"].cpu().numpy(),
                        "labels": target["labels"].cpu().numpy(),
                    })
        metrics = _evaluate_detections(predictions, ground_truth, confidence_threshold=confidence_threshold)
        map_value = metrics["map50_95_macro_over_classes_with_ground_truth"]
        history.append({
            "epoch": epoch,
            "mean_training_loss": sum(epoch_losses) / max(len(epoch_losses), 1),
            "validation_metrics": metrics,
            "epoch_seconds": round(time.time() - started, 2),
        })
        print(
            f"epoch {epoch} complete: loss={history[-1]['mean_training_loss']:.4f}; "
            f"mAP50={metrics['map50_macro_over_classes_with_ground_truth']}; "
            f"mAP50:95={map_value}; {history[-1]['epoch_seconds']}s",
            flush=True,
        )
        if map_value is not None and map_value > best_map:
            best_map = map_value
            best_epoch = epoch
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint = output_root / "fasterrcnn_mobilenetv3_rdd2022_india.pt"
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "architecture": "torchvision fasterrcnn_mobilenet_v3_large_320_fpn",
        "class_to_id": LABEL_TO_ID,
        "id_to_class": ID_TO_LABEL,
        "seed": seed,
        "best_validation_map50_95": best_map if best_map >= 0 else None,
    }, checkpoint)
    selected_metrics = best_metrics or history[-1]["validation_metrics"]
    report = {
        "project": "India road-crash hotspot and hazard analysis — CV road-surface detector",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_doi": FIGSHARE_DOI,
        "dataset_manifest": str((raw_root / "source_manifest.json").resolve()),
        "model": {
            "architecture": "Torchvision Faster R-CNN MobileNetV3-Large 320 FPN",
            "pretrained_on_coco": not no_pretrained,
            "fine_tuned_classes": ID_TO_LABEL,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "device": str(device),
            "checkpoint": str(checkpoint.resolve()),
        },
        "evaluation": {
            "split_file": str(split_path.resolve()),
            "selected_run_split_file": str((processed_root / "selected_run_split.json").resolve()),
            "split_type": "deterministic holdout by blocks of sequential numeric image IDs (proxy only)",
            "is_verified_geographic_holdout": False,
            "train_images_used": len(train_ds),
            "validation_images_used": len(val_ds),
            "training_target_instance_counts": dict(selected_training_class_counts),
            "validation_target_instance_counts": dict(selected_validation_class_counts),
            "full_grouped_train_images_available": available_train_images,
            "full_grouped_validation_images_available": available_validation_images,
            "zero_target_train_images_excluded_to_avoid_nonfinite_empty_only_batches": (
                train_background_images_excluded
            ),
            "clean_zero_target_validation_images_retained": sum(
                record["image_id"] in val_ids and record["object_count"] == 0 for record in records
            ),
            "selected_checkpoint_epoch": best_epoch or epochs,
            "seed": seed,
            "confidence_threshold": confidence_threshold,
            "selected_checkpoint_metrics": selected_metrics,
            "epoch_history": history,
        },
        "interpretation_limits": [
            "Metrics describe this held-out subset of annotated RDD2022 India training images only.",
            "The holdout groups are filename-sequence proxies; source files contain no verified location grouping.",
            "The challenge test images have no public ground truth and were not scored.",
            "RDD2022 image locations/timestamps are not linked to the separate crash events in this project.",
            "Detection is a surface-condition cue for human inspection, not evidence that a road defect caused a crash.",
        ],
    }
    _json_write(output_root / "metrics.json", report)
    return report


def predict_images(raw_image_paths: Iterable[Path], checkpoint: Path, output_path: Path,
                   confidence_threshold: float) -> dict[str, Any]:
    torch, Image, _, _, model_components = _torch_components()
    torchvision, FastRCNNPredictor = model_components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(torchvision, FastRCNNPredictor, pretrained=False)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device).eval()
    results = []
    with torch.inference_mode():
        for image_path in raw_image_paths:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                tensor = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255
            prediction = model([tensor.to(device)])[0]
            rows = []
            for box, label, score in zip(
                prediction["boxes"].cpu().tolist(),
                prediction["labels"].cpu().tolist(),
                prediction["scores"].cpu().tolist(),
            ):
                if score >= confidence_threshold:
                    class_code = next(
                        (code for code in CLASSES if LABEL_TO_ID[code] == label), None
                    )
                    rows.append({
                        "class_id": int(label),
                        "class_code": class_code,
                        "class_name": CLASSES.get(class_code) if class_code else None,
                        "confidence": float(score),
                        "xyxy": [float(value) for value in box],
                    })
            results.append({"image": str(image_path.resolve()), "detections": rows})
    payload = {
        "checkpoint": str(checkpoint.resolve()),
        "confidence_threshold": confidence_threshold,
        "interpretation": "Road condition detections require human review and are not crash causation labels.",
        "predictions": results,
    }
    _json_write(output_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RDD2022 India road-damage CV pipeline")
    parser.add_argument("--raw-root", type=Path, default=_default_raw_root())
    parser.add_argument("--processed-root", type=Path, default=_default_processed_root())
    subs = parser.add_subparsers(dest="command", required=True)

    subs.add_parser("remote-index", help="inspect Figshare ZIP metadata via range reads only")
    download_parser = subs.add_parser(
        "download-india", help="range-extract India/train JPEG and VOC XML members only"
    )
    download_parser.add_argument("--max-compressed-gb", type=float, default=1.5)

    profile_parser = subs.add_parser("profile", help="validate local VOC labels and count classes")
    profile_parser.add_argument("--group-size", type=int, default=100)

    split_parser = subs.add_parser("split", help="save sequence-proxy grouped train/validation IDs")
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--validation-fraction", type=float, default=0.2)
    split_parser.add_argument("--group-size", type=int, default=100)

    train_parser = subs.add_parser("train", help="fine-tune and evaluate compact Faster R-CNN")
    train_parser.add_argument("--output-dir", type=Path, default=_default_processed_root())
    train_parser.add_argument(
        "--full-run", action="store_true",
        help="use the full grouped India train/validation split; default is a 1-epoch capped run",
    )
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--learning-rate", type=float, default=0.00005)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--max-train-images", type=int, default=None)
    train_parser.add_argument("--max-val-images", type=int, default=None)
    train_parser.add_argument("--confidence-threshold", type=float, default=0.5)
    train_parser.add_argument("--no-pretrained", action="store_true",
                              help="do not download/use COCO initialization weights")

    predict_parser = subs.add_parser("predict", help="run the trained detector on local images")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--confidence-threshold", type=float, default=0.5)
    predict_parser.add_argument("images", nargs="+", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.processed_root.mkdir(parents=True, exist_ok=True)
    if args.command == "remote-index":
        result = remote_archive_index(args.processed_root)
    elif args.command == "download-india":
        result = extract_india_training(args.raw_root, args.processed_root, args.max_compressed_gb)
    elif args.command == "profile":
        result = profile_local_data(args.raw_root, args.processed_root)
        rows = json.loads((args.processed_root / "voc_records.json").read_text(encoding="utf-8"))
        eligible_rows = [row for row in rows if not row.get("out_of_scope_class_codes")]
        split = create_grouped_split(eligible_rows, group_size=args.group_size)
        split["images_excluded_for_out_of_scope_labels"] = len(rows) - len(eligible_rows)
        split["dataset_profile"] = result
        _json_write(args.processed_root / "grouped_split.json", split)
    elif args.command == "split":
        result = build_split_file(args.raw_root, args.processed_root, args.seed,
                                  args.validation_fraction, args.group_size)
    elif args.command == "train":
        epochs = args.epochs if args.epochs is not None else (10 if args.full_run else 1)
        max_train_images = args.max_train_images
        max_val_images = args.max_val_images
        if not args.full_run:
            max_train_images = max_train_images or 1200
            max_val_images = max_val_images or 300
        result = train_detector(
            args.raw_root, args.processed_root, args.output_dir,
            epochs=epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            max_train_images=max_train_images,
            max_val_images=max_val_images,
            confidence_threshold=args.confidence_threshold,
            no_pretrained=args.no_pretrained,
        )
    elif args.command == "predict":
        result = predict_images(args.images, args.checkpoint, args.output, args.confidence_threshold)
    else:
        raise ValueError(f"unsupported command: {args.command}")
    printable = result
    if args.command == "split":
        printable = {
            key: value for key, value in result.items()
            if key not in {"train_image_ids", "validation_image_ids", "dataset_profile"}
        }
        printable["train_image_count"] = len(result["train_image_ids"])
        printable["validation_image_count"] = len(result["validation_image_ids"])
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
