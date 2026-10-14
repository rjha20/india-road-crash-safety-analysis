"""Privacy-conscious, descriptive NLP for the local 2022 Mendeley article corpus.

The source contains newspaper prose.  This module reads it locally and writes
only corpus-level theme counts/metadata; it never serializes article text or
article-level predictions.  Rule matches are weak mention labels, not verified
causes, truth labels, or measured NLP accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = PROJECT_ROOT / "data/raw/mendeley/v7/raw_articles/2022_raw_data.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed"

# Broad expressions are deliberately treated as *mentions*.  They do not
# resolve negation, whose/which-vehicle attribution, article framing, or cause.
THEMES: dict[str, tuple[str, ...]] = {
    "speed_or_unsafe_manoeuvre": (
        r"\b(over\s*speed(?:ing)?|speeding|rash driving|reckless driving|high speed|overspeed(?:ing)?)\b",
    ),
    "alcohol_mentioned": (r"\b(alcohol|drunk(?:en)? driving|under the influence|liquor)\b",),
    "helmet_mentioned": (r"\b(helmet|helmets)\b",),
    "seat_belt_mentioned": (r"\b(seat[ -]?belt|seatbelt)\b",),
    "wrong_side_or_lane": (r"\b(wrong[- ]side|wrong[- ]way|wrong lane|against the traffic)\b",),
    "road_surface_or_design": (
        r"\b(pothole|potholes|road condition|poor road|damaged road|road defect|sharp turn|blind curve)\b",
    ),
    "weather_or_visibility": (
        r"\b(fog|dense fog|rain(?:fall)?|heavy rain|low visibility|poor visibility|smoke|mist)\b",
    ),
    "hit_and_run_mentioned": (r"\b(hit[- ]and[- ]run|fled the scene|fled from the spot)\b",),
    "pedestrian_or_cyclist_mentioned": (r"\b(pedestrian|pedestrians|cyclist|cyclists|bicycle)\b",),
    "two_wheeler_mentioned": (r"\b(motorcycle|motorbike|two[- ]wheeler|scooter|bike)\b",),
    "heavy_vehicle_mentioned": (r"\b(truck|lorry|bus|trailer|heavy vehicle)\b",),
}


def _column_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _detect(columns: list[str], choices: tuple[str, ...]) -> str | None:
    by_key = {_column_key(col): col for col in columns}
    for choice in choices:
        if _column_key(choice) in by_key:
            return by_key[_column_key(choice)]
    return None


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value)).casefold()


def analyze_articles(raw_path: Path = DEFAULT_RAW, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"2022 raw-article CSV not found: {raw_path}. Download the dataset's Raw Data/2022 file."
        )
    # The failed 2023 request was a short JSON error, not a valid CSV.  This
    # guard prevents accidentally loading it as a one-row article corpus.
    if raw_path.stat().st_size < 1024:
        head = raw_path.read_bytes()[:512].decode("utf-8", errors="replace").lstrip()
        if head.startswith("{") or head.startswith("["):
            raise ValueError("Input looks like a small JSON error response, not the article CSV.")

    raw = pd.read_csv(raw_path, encoding="utf-8-sig", low_memory=False)
    source_row_count = int(len(raw))
    raw.columns = [str(c).strip() for c in raw.columns]
    title_col = _detect(list(raw.columns), ("Title", "Article Title"))
    date_col = _detect(list(raw.columns), ("Date", "Article Date", "Publication Date"))
    content_col = _detect(list(raw.columns), ("Content", "Article Content", "Text", "Body"))
    if not content_col:
        raise ValueError(
            f"No article content column found in {raw_path.name}; observed columns are {list(raw.columns)!r}."
        )

    title = raw[title_col].map(_safe_text) if title_col else pd.Series("", index=raw.index)
    content = raw[content_col].map(_safe_text)
    if date_col:
        # The source uses the literal abbreviation IST, which pandas does not
        # treat as an unambiguous timezone. The analysis needs the article date,
        # so strip only that terminal display suffix before parsing.
        date_text = raw[date_col].astype("string")
        # Some rows begin with the publisher's `Updated:` label. It is a
        # formatting prefix, not part of the timestamp.
        date_text = date_text.str.replace(r"^Updated:\s*", "", case=False, regex=True)
        date_text = date_text.str.replace(r"\s+(IST|UTC)$", "", case=False, regex=True)
        dates = pd.to_datetime(date_text, errors="coerce", dayfirst=False, format="mixed")
    else:
        dates = pd.Series(pd.NaT, index=raw.index)
    text = (title + " " + content).str.strip()
    nonempty = text.str.len().gt(0)
    raw, text, dates = raw.loc[nonempty].copy(), text.loc[nonempty], dates.loc[nonempty]

    # Deduplicate exact article bodies before computing counts.  Hashes remain
    # in memory only and are not exported.
    text_hashes = text.map(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())
    duplicate_mask = text_hashes.duplicated(keep="first")
    text, dates = text.loc[~duplicate_mask], dates.loc[~duplicate_mask]
    docs = text.tolist()
    counts = []
    compiled = {name: re.compile("|".join(parts), re.IGNORECASE) for name, parts in THEMES.items()}
    for theme, pattern in compiled.items():
        mentions = sum(bool(pattern.search(doc)) for doc in docs)
        counts.append({
            "theme": theme,
            "articles_with_mention": int(mentions),
            "article_share_pct": round(mentions / len(docs) * 100, 3) if docs else 0.0,
            "label_type": "weak keyword mention; not gold annotation or cause",
        })
    theme_table = pd.DataFrame(counts).sort_values("articles_with_mention", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    theme_table.to_csv(output_dir / "nlp_theme_mentions_2022.csv", index=False)
    summary: dict[str, Any] = {
        "source_file": raw_path.name,
        "source_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "source_year_scope": "2022 raw articles only",
        "source_rows": source_row_count,
        "usable_nonempty_articles": int(len(docs) + int(duplicate_mask.sum())),
        "exact_duplicate_articles_removed_for_counts": int(duplicate_mask.sum()),
        "unique_articles_analyzed": int(len(docs)),
        "title_column_detected": title_col,
        "article_date_column_detected": date_col,
        "content_column_detected": content_col,
        "articles_with_unparsed_or_missing_article_date": int(dates.isna().sum()),
        "article_date_min": None if dates.dropna().empty else dates.min().strftime("%Y-%m-%d"),
        "article_date_max": None if dates.dropna().empty else dates.max().strftime("%Y-%m-%d"),
        "theme_mentions": {row.theme: {"articles": int(row.articles_with_mention), "share_pct": float(row.article_share_pct)} for row in theme_table.itertuples(index=False)},
        "limitations": [
            "Only the accessible 2022 raw-article file was analyzed; the 2023 request produced a 395-byte internal-error JSON response and was excluded.",
            "The 2022 prose is a media-selected corpus of fatal-crash reporting, not all crashes or a representative population sample.",
            "Keyword themes are weak mention indicators. Negation, attribution, chronology, sarcasm, and event causation are not resolved.",
            "No manually adjudicated gold sample is available; precision, recall, F1, and model accuracy are not reported.",
            "Article bodies, titles, names, and article-level tags are not written to processed artifacts; raw text remains in the local source folder.",
        ],
    }
    (output_dir / "nlp_summary_2022.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = analyze_articles(args.input, args.output_dir)
    print(json.dumps({
        "source_year_scope": summary["source_year_scope"],
        "unique_articles_analyzed": summary["unique_articles_analyzed"],
        "article_date_min": summary["article_date_min"],
        "article_date_max": summary["article_date_max"],
        "top_themes": sorted(summary["theme_mentions"].items(), key=lambda kv: kv[1]["articles"], reverse=True)[:5],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
