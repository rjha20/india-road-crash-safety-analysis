"""Load, normalize, and quality-check the Mendeley v7 structured crash workbook.

The source is a selected set of fatal crashes reported by one newspaper.  The
normalized file is an analysis aid; it is not a census of Indian crashes and
its counts must not be described as road risk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data/raw/mendeley/v7/news_crashes.xlsx"
DEFAULT_DICTIONARY = PROJECT_ROOT / "data/raw/mendeley/v7/data_dictionary.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed"
DEFAULT_REPORT = PROJECT_ROOT / "reports/data_quality.md"

REQUIRED_COLUMNS = {
    "S. No.",
    "Month",
    "Crash Date",
    "Crash Day",
    "Article Date",
    "Location",
    "Million Plus City",
    "State",
    "LatLong",
    "Vehicle 1",
    "Vehicle/Object 2",
    "Killed",
    "Injured",
    "Age",
    "Gender",
    "Road Type",
    "Crash Type",
}

MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "unknown", "not reported", "-"}
INDIA_BOUNDS = (6.0, 38.0, 67.0, 98.0)  # broad QA envelope, not a geocoder
PUBLIC_GRID_CELL_DEGREES = 0.25
PUBLIC_GRID_MIN_SUPPORT = 5


@dataclass
class CrashDataResult:
    events: pd.DataFrame
    quality: dict[str, Any]
    missingness: pd.DataFrame
    duplicate_rows: pd.DataFrame
    dictionary: pd.DataFrame
    source_sha256: str


def build_public_event_grid(
    events: pd.DataFrame,
    *,
    cell_degrees: float = PUBLIC_GRID_CELL_DEGREES,
    min_support: int = PUBLIC_GRID_MIN_SUPPORT,
) -> pd.DataFrame:
    """Build a shareable coarse grid without publishing row-level event records.

    Each released slice is grouped by year, state, and 0.25-degree cell.
    Groups below the minimum support are suppressed before writing.
    Article text, dates, exact coordinates, and source row identifiers are
    deliberately excluded.
    """
    if cell_degrees <= 0 or min_support < 1:
        raise ValueError("cell_degrees and min_support must be positive")
    required = {"crash_date", "state", "latitude", "longitude", "killed"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing columns for public grid: {', '.join(sorted(missing))}")

    frame = events.loc[:, sorted(required)].copy()
    frame["crash_date"] = pd.to_datetime(frame["crash_date"], errors="coerce")
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame["killed"] = pd.to_numeric(frame["killed"], errors="coerce")
    frame = frame.dropna(subset=["crash_date", "latitude", "longitude"])
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "year", "state", "cell_latitude", "cell_longitude",
                "reported_events", "recorded_fatalities",
            ]
        )

    frame["year"] = frame["crash_date"].dt.year.astype(int)
    frame["state"] = frame["state"].astype("string").fillna("Unknown")
    frame["state"] = frame["state"].astype("string").str.strip().replace("", "Unknown").fillna("Unknown")
    frame["cell_latitude"] = frame["latitude"].map(
        lambda value: (math.floor(float(value) / cell_degrees) + 0.5) * cell_degrees
    )
    frame["cell_longitude"] = frame["longitude"].map(
        lambda value: (math.floor(float(value) / cell_degrees) + 0.5) * cell_degrees
    )

    group_columns = ["year", "state", "cell_latitude", "cell_longitude"]
    public_grid = (
        frame.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            reported_events=("crash_date", "size"),
            recorded_fatalities=("killed", lambda values: values.sum(min_count=1)),
        )
    )
    public_grid = public_grid[public_grid["reported_events"] >= min_support].copy()
    public_grid["cell_latitude"] = public_grid["cell_latitude"].round(3)
    public_grid["cell_longitude"] = public_grid["cell_longitude"].round(3)
    return public_grid.sort_values(group_columns, kind="stable").reset_index(drop=True)


def _header(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _clean_text(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    text = re.sub(r"\s+", " ", str(value)).strip()
    if text.casefold() in MISSING_TOKENS:
        return pd.NA
    return text


def _parse_latlong(value: Any) -> tuple[float | None, float | None]:
    if pd.isna(value):
        return None, None
    # The dictionary defines LatLong as a tuple; the workbook stores strings
    # such as "18.2, 76.0".  Do not guess coordinate order when it is malformed.
    parts = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)", str(value))
    if len(parts) != 2:
        return None, None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None, None


def _date_value(value: Any) -> pd.Timestamp | pd.NaT:
    if pd.isna(value):
        return pd.NaT
    return pd.to_datetime(value, errors="coerce", dayfirst=False)


def _dict_for_report(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["Variable Name", "Data Type", "Description"])
    return pd.read_excel(path, sheet_name=0).dropna(how="all")


def load_and_profile(source: Path = DEFAULT_SOURCE, dictionary_path: Path = DEFAULT_DICTIONARY) -> CrashDataResult:
    if not source.exists():
        raise FileNotFoundError(
            f"Mendeley workbook not found at {source}. Download v7 Data Files and place "
            "news_crashes.xlsx and data_dictionary.xlsx in data/raw/mendeley/v7/."
        )

    raw = pd.read_excel(source, sheet_name=0)
    raw.columns = [_header(c) for c in raw.columns]
    missing_columns = sorted(REQUIRED_COLUMNS.difference(raw.columns))
    if missing_columns:
        raise ValueError(
            "The workbook schema differs from the inspected Mendeley v7 workbook; "
            f"missing expected columns: {missing_columns}. Inspect the source dictionary before mapping."
        )
    raw = raw[list(REQUIRED_COLUMNS)]

    original = raw.copy()
    # Nil in Million Plus City is a source value meaning no million-plus city was
    # supplied; retain it as an explicit category rather than treating it as NA.
    text_columns = [
        "Month", "Crash Day", "Location", "Million Plus City", "State", "Vehicle 1",
        "Vehicle/Object 2", "Age", "Gender", "Road Type", "Crash Type",
    ]
    for col in text_columns:
        raw[col] = raw[col].map(_clean_text)
    city = raw["Million Plus City"].astype("string").str.casefold().eq("nil")
    raw.loc[city, "Million Plus City"] = "Not listed as a million-plus city"

    raw["Crash Date"] = raw["Crash Date"].map(_date_value)
    raw["Article Date"] = raw["Article Date"].map(_date_value)
    raw["Killed"] = pd.to_numeric(raw["Killed"], errors="coerce")
    raw["Injured"] = pd.to_numeric(raw["Injured"], errors="coerce")
    raw["S. No."] = pd.to_numeric(raw["S. No."], errors="coerce")

    coordinates = raw["LatLong"].map(_parse_latlong)
    raw["latitude"] = coordinates.map(lambda pair: pair[0])
    raw["longitude"] = coordinates.map(lambda pair: pair[1])
    lat_min, lat_max, lon_min, lon_max = INDIA_BOUNDS
    raw["coordinate_in_india_envelope"] = (
        raw["latitude"].between(lat_min, lat_max)
        & raw["longitude"].between(lon_min, lon_max)
    )

    crash_month = raw["Crash Date"].dt.month_name()
    reported_month = raw["Month"].astype("string").str.casefold()
    month_mismatch = crash_month.astype("string").str.casefold().ne(reported_month)
    month_mismatch &= raw["Crash Date"].notna() & raw["Month"].notna()
    weekday = raw["Crash Date"].dt.day_name()
    day_mismatch = weekday.astype("string").str.casefold().ne(
        raw["Crash Day"].astype("string").str.casefold()
    )
    day_mismatch &= raw["Crash Date"].notna() & raw["Crash Day"].notna()

    # Exact repeated crash signatures are surfaced, not deleted.  No fuzzy
    # merge is attempted because it can collapse distinct crashes in the same
    # locality/date.
    duplicate_key = pd.DataFrame({
        "date": raw["Crash Date"].dt.strftime("%Y-%m-%d"),
        "state": raw["State"].astype("string").str.casefold().str.strip(),
        "location": raw["Location"].astype("string").str.casefold().str.strip(),
        "killed": raw["Killed"],
        "injured": raw["Injured"],
        "vehicle_1": raw["Vehicle 1"].astype("string").str.casefold().str.strip(),
        "vehicle_2": raw["Vehicle/Object 2"].astype("string").str.casefold().str.strip(),
    })
    dup_mask = duplicate_key.duplicated(keep=False)
    duplicate_rows = raw.loc[dup_mask, ["S. No.", "Crash Date", "State", "Location", "Killed", "Injured"]].copy()
    duplicate_rows = duplicate_rows.rename(columns={"S. No.": "source_row_number"})

    # Source-level completeness includes the two person-description fields,
    # but their underlying values are not copied into the processed event file.
    missing_records = []
    for col in original.columns:
        source_series = original[col]
        if source_series.dtype == "object" or pd.api.types.is_string_dtype(source_series):
            semantic_missing = source_series.isna() | source_series.astype("string").str.strip().str.casefold().isin(MISSING_TOKENS)
            if col == "Million Plus City":
                semantic_missing &= ~source_series.astype("string").str.strip().str.casefold().eq("nil")
        else:
            semantic_missing = source_series.isna()
        missing_records.append({
            "source_column": col,
            "missing_count": int(semantic_missing.sum()),
            "missing_pct": float(semantic_missing.mean() * 100),
        })
    missingness = pd.DataFrame(missing_records).sort_values(
        "missing_pct", ascending=False, kind="stable"
    )

    # Do not export age or gender lists: they can contain identifiable victim
    # attributes and are not used by the model.
    events = pd.DataFrame({
        "source_row_number": raw["S. No."].astype("Int64"),
        "crash_date": raw["Crash Date"].dt.strftime("%Y-%m-%d"),
        "article_date": raw["Article Date"].dt.strftime("%Y-%m-%d"),
        "source_month": raw["Month"],
        "source_crash_day": raw["Crash Day"],
        "location": raw["Location"],
        "million_plus_city": raw["Million Plus City"],
        "state": raw["State"],
        "latitude": raw["latitude"],
        "longitude": raw["longitude"],
        "coordinate_in_india_envelope": raw["coordinate_in_india_envelope"],
        "vehicle_1": raw["Vehicle 1"],
        "vehicle_2": raw["Vehicle/Object 2"],
        "killed": raw["Killed"],
        "injured": raw["Injured"],
        "road_type": raw["Road Type"],
        "crash_type": raw["Crash Type"],
    })

    sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    quality: dict[str, Any] = {
        "source_file": source.name,
        "source_sha256": sha256,
        "rows": int(len(raw)),
        "physical_columns": int(len(original.columns)),
        "dictionary_variables_excluding_serial": int(len(original.columns) - 1),
        "source_row_id_unique": bool(raw["S. No."].is_unique),
        "source_row_id_missing": int(raw["S. No."].isna().sum()),
        "exact_duplicate_rows_including_serial": int(original.duplicated().sum()),
        "duplicate_signature_rows_flagged": int(dup_mask.sum()),
        "duplicate_signature_groups": int(len(duplicate_key.loc[dup_mask].drop_duplicates())),
        "crash_date_min": None if raw["Crash Date"].dropna().empty else raw["Crash Date"].min().strftime("%Y-%m-%d"),
        "crash_date_max": None if raw["Crash Date"].dropna().empty else raw["Crash Date"].max().strftime("%Y-%m-%d"),
        "article_date_min": None if raw["Article Date"].dropna().empty else raw["Article Date"].min().strftime("%Y-%m-%d"),
        "article_date_max": None if raw["Article Date"].dropna().empty else raw["Article Date"].max().strftime("%Y-%m-%d"),
        "crash_month_disagreement_count": int(month_mismatch.sum()),
        "crash_weekday_disagreement_count": int(day_mismatch.sum()),
        "coordinates_parsed_count": int(raw["latitude"].notna().sum()),
        "coordinates_in_india_envelope_count": int(raw["coordinate_in_india_envelope"].sum()),
        "coordinates_outside_envelope_count": int((raw["latitude"].notna() & ~raw["coordinate_in_india_envelope"]).sum()),
        "coordinates_unparsed_count": int(raw["latitude"].isna().sum()),
        "states": int(raw["State"].nunique(dropna=True)),
        "fatality_min": None if raw["Killed"].dropna().empty else float(raw["Killed"].min()),
        "fatality_max": None if raw["Killed"].dropna().empty else float(raw["Killed"].max()),
        "total_fatalities_in_sample": int(raw["Killed"].sum(min_count=1)) if raw["Killed"].notna().any() else None,
        "total_injuries_in_sample": int(raw["Injured"].sum(min_count=1)) if raw["Injured"].notna().any() else None,
        "fatality_band_counts": {
            "one_fatality": int(raw["Killed"].eq(1).sum()),
            "two_or_more_fatalities": int(raw["Killed"].ge(2).sum()),
            "missing_or_invalid": int(raw["Killed"].isna().sum()),
        },
        "missingness": {row.source_column: {"count": int(row.missing_count), "pct": round(float(row.missing_pct), 4)} for row in missingness.itertuples(index=False)},
        "privacy_note": "Age/Gender source values are not copied to processed crash_events_clean.csv.",
        "interpretation_note": "Media-selected fatal crashes; not representative of all crashes and not a risk-rate denominator.",
    }
    return CrashDataResult(events, quality, missingness, duplicate_rows, _dict_for_report(dictionary_path), sha256)


def write_outputs(result: CrashDataResult, output_dir: Path = DEFAULT_OUTPUT, report_path: Path = DEFAULT_REPORT) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    public_grid = build_public_event_grid(result.events)
    result.events.to_csv(output_dir / "crash_events_clean.csv", index=False)
    public_grid.to_csv(output_dir / "reported_event_grid_public.csv", index=False)
    result.missingness.to_csv(output_dir / "crash_missingness.csv", index=False)
    result.duplicate_rows.to_csv(output_dir / "crash_duplicate_signatures.csv", index=False)
    result.dictionary.to_csv(output_dir / "crash_data_dictionary.csv", index=False)
    (output_dir / "crash_quality_summary.json").write_text(
        json.dumps(result.quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    q = result.quality
    q["public_grid_cells"] = int(len(public_grid))
    q["public_grid_events"] = int(public_grid["reported_events"].sum()) if not public_grid.empty else 0
    q["public_grid_cell_min_support"] = PUBLIC_GRID_MIN_SUPPORT
    miss_lines = [
        f"| `{row.source_column}` | {row.missing_count:,} | {row.missing_pct:.2f}% |"
        for row in result.missingness.itertuples(index=False)
    ]
    report = f"""# Crash data quality and coverage

## Source and scope

- **Source:** Ashutosh and Chand, *Media Reported Road Traffic Crash Data*, Mendeley Data v7, DOI [10.17632/bc5sv6wnd9.7](https://doi.org/10.17632/bc5sv6wnd9.7), CC BY 4.0 for the dataset. The structured workbook was downloaded from the v7 Data Files folder.
- **Input file:** `{q['source_file']}`; SHA-256 `{q['source_sha256']}`.
- **Observed table:** {q['rows']:,} records and {q['physical_columns']} physical columns (serial number plus {q['dictionary_variables_excluding_serial']} dictionary variables).
- **Population represented:** fatal crashes reported by The Times of India, not all Indian crashes. The data is editorially selected and has no traffic-exposure denominator; event totals and mapped concentrations are not comparative road-risk estimates.
- **Narratives:** this structured workbook contains no article title or body. Raw narrative analysis is restricted to the separate 2022 article file; 2023 raw-file access returned an error and is excluded.

## Completeness

Missingness uses empty/NA-like tokens. The source value `Nil` for Million Plus City is retained as “Not listed as a million-plus city,” not interpreted as an unknown value.

| Source field | Missing | Missing % |
|---|---:|---:|
{chr(10).join(miss_lines)}

`Age` and `Gender` have source missingness, and may include multi-victim values. Their source values were deliberately omitted from processed event rows because they are not needed for this project’s model.

## Integrity and geocoding checks

- Source serial number unique: **{q['source_row_id_unique']}**; missing serials: {q['source_row_id_missing']}.
- Exact duplicate rows including the serial: {q['exact_duplicate_rows_including_serial']}.
- Rows flagged by a conservative duplicate signature (date, state, location, fatalities, injuries, and both vehicle fields): {q['duplicate_signature_rows_flagged']} across {q['duplicate_signature_groups']} signature group(s). These rows are surfaced in `data/processed/crash_duplicate_signatures.csv`, not silently removed. The signature can flag distinct events sharing a place/date, so it requires review.
- Crash-date coverage: {q['crash_date_min']} through {q['crash_date_max']}; article-date coverage: {q['article_date_min']} through {q['article_date_max']}.
- Source month disagrees with parsed crash-date month on {q['crash_month_disagreement_count']:,} rows; weekday disagrees on {q['crash_weekday_disagreement_count']:,} rows. Dates were not “corrected” from those secondary text fields.
- Coordinates parsed: {q['coordinates_parsed_count']:,}/{q['rows']:,}; within a broad India envelope: {q['coordinates_in_india_envelope_count']:,}; outside envelope: {q['coordinates_outside_envelope_count']:,}; unparsed: {q['coordinates_unparsed_count']:,}. This range check is not a positional accuracy audit or road snap.
- States represented: {q['states']}.

## Outcome counts and cautions

- Fatalities recorded in the selected sample: {q['total_fatalities_in_sample']:,}; injuries: {q['total_injuries_in_sample']:,}.
- Rows with one recorded death: {q['fatality_band_counts']['one_fatality']:,}; rows with two or more: {q['fatality_band_counts']['two_or_more_fatalities']:,}.
- Every structured row belongs to a **fatal-crash sample**. Modeling the number of deaths is therefore a conditional description among reported fatal crashes, not prediction of whether a road will have a crash, not national crash risk, and not a causal explanation.
- Coordinates/location labels were extracted from reports. The envelope check cannot establish geocoding precision; a road-network join still requires manual verification and match-confidence reporting.

## Reproduction and output handling

Run `python -m road_safety.crash_data` from the project environment. The row-level `crash_events_clean.csv` remains local and gitignored. The shareable `reported_event_grid_public.csv` contains only year/state/coarse-cell counts and fatalities; cells with fewer than {PUBLIC_GRID_MIN_SUPPORT} sampled records per year/state slice are suppressed. It omits exact coordinates, dates, location descriptions, article text, and source row identifiers. Other processed CSV/JSON outputs are generated under `data/processed/`. The pipeline omits victim Age/Gender values and all raw news article text from its processed files.
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = load_and_profile(args.source, args.dictionary)
    write_outputs(result, args.output_dir, args.report)
    print(json.dumps({k: result.quality[k] for k in (
        "rows", "crash_date_min", "crash_date_max", "coordinates_parsed_count",
        "duplicate_signature_rows_flagged", "crash_month_disagreement_count",
        "crash_weekday_disagreement_count", "fatality_band_counts",
    )}, indent=2))


if __name__ == "__main__":
    main()
