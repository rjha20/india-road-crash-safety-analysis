"""Exploratory spatial concentration analysis for the media-reported sample.

Outputs are deliberately aggregated to coarse cells and cluster summaries. The
source is a fatal-crash news sample, so these outputs describe reported-event
concentrations only; they are not road risk, crash incidence, or engineering
black-spot designations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data/processed/crash_events_clean.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed"
DEFAULT_REPORT = PROJECT_ROOT / "reports/spatial_hotspot_analysis.md"
INDIA_BOUNDS = (6.0, 38.0, 67.0, 98.0)
EARTH_RADIUS_KM = 6371.0088


def _candidate_signature_masks(events: pd.DataFrame) -> tuple[pd.Series, pd.Series, int]:
    """Return full duplicate-group flags, removable repeats, and group count."""
    columns = ["crash_date", "state", "location", "killed", "injured", "vehicle_1", "vehicle_2"]
    missing = set(columns).difference(events.columns)
    if missing:
        raise ValueError(f"Input lacks signature columns: {sorted(missing)}")
    key = events[columns].copy()
    for name in ("state", "location", "vehicle_1", "vehicle_2"):
        key[name] = key[name].astype("string").str.casefold().str.strip()
    in_duplicate_group = key.duplicated(keep=False)
    repeated_after_first = key.duplicated(keep="first")
    group_count = int(key.loc[in_duplicate_group].drop_duplicates().shape[0])
    return in_duplicate_group, repeated_after_first, group_count


def _dbscan_summary(points: pd.DataFrame, *, eps_km: float, min_samples: int) -> dict[str, Any]:
    if points.empty:
        return {"eps_km": eps_km, "min_samples": min_samples, "cluster_count": 0,
                "clustered_events": 0, "noise_events": 0, "clusters": []}
    radians = np.radians(points[["latitude", "longitude"]].to_numpy(dtype=float))
    labels = DBSCAN(eps=eps_km / EARTH_RADIUS_KM, min_samples=min_samples,
                    metric="haversine", algorithm="ball_tree").fit_predict(radians)
    rows: list[dict[str, Any]] = []
    for label in sorted(set(labels).difference({-1})):
        group = points.loc[labels == label]
        # Round centers to 0.1 degrees (~11 km latitude); never export event points.
        rows.append({
            "cluster": int(label),
            "events": int(len(group)),
            "center_latitude_rounded_0_1deg": round(float(group.latitude.median()), 1),
            "center_longitude_rounded_0_1deg": round(float(group.longitude.median()), 1),
            "states": int(group.state.nunique()),
            "date_min": str(group.crash_date.min().date()) if group.crash_date.notna().any() else None,
            "date_max": str(group.crash_date.max().date()) if group.crash_date.notna().any() else None,
        })
    rows.sort(key=lambda row: (-row["events"], row["cluster"]))
    return {
        "eps_km": eps_km,
        "min_samples": min_samples,
        "cluster_count": len(rows),
        "clustered_events": int((labels >= 0).sum()),
        "noise_events": int((labels == -1).sum()),
        "clusters": rows,
    }


def analyze_spatial(events_path: Path, output_dir: Path, report_path: Path,
                    *, cell_degrees: float = 0.25, eps_km: float = 25.0,
                    min_samples: int = 5) -> dict[str, Any]:
    events = pd.read_csv(events_path, parse_dates=["crash_date"])
    required = {"latitude", "longitude", "state", "crash_date", "location", "killed", "injured",
                "vehicle_1", "vehicle_2"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Input lacks required spatial fields: {sorted(missing)}")
    lat = pd.to_numeric(events.latitude, errors="coerce")
    lon = pd.to_numeric(events.longitude, errors="coerce")
    south, north, west, east = INDIA_BOUNDS
    valid = lat.between(south, north) & lon.between(west, east)
    points = events.loc[valid].copy()
    points["latitude"] = lat.loc[valid]
    points["longitude"] = lon.loc[valid]
    if points.empty:
        raise ValueError("No coordinates passed the broad India QA envelope")

    duplicate_group, duplicate_candidate, duplicate_group_count = _candidate_signature_masks(points)
    points["candidate_duplicate_group"] = duplicate_group.to_numpy()
    points["candidate_repeated_row"] = duplicate_candidate.to_numpy()
    deduped = points.loc[~points.candidate_repeated_row].copy()

    scale = 1.0 / cell_degrees
    points["lat_bin"] = np.floor(points.latitude * scale).astype(int)
    points["lon_bin"] = np.floor(points.longitude * scale).astype(int)
    deduped["lat_bin"] = np.floor(deduped.latitude * scale).astype(int)
    deduped["lon_bin"] = np.floor(deduped.longitude * scale).astype(int)
    group_cols = ["lat_bin", "lon_bin"]
    all_counts = points.groupby(group_cols, observed=True).size().rename("reported_events").to_frame()
    dedup_counts = deduped.groupby(group_cols, observed=True).size().rename("after_candidate_dedup").to_frame()
    cells = all_counts.join(dedup_counts, how="outer").fillna(0).astype(int).reset_index()
    cells["cell_latitude"] = ((cells.lat_bin + 0.5) * cell_degrees).round(3)
    cells["cell_longitude"] = ((cells.lon_bin + 0.5) * cell_degrees).round(3)
    cells["cell_width_km_approx_at_center"] = (
        111.32 * cell_degrees * np.cos(np.radians(cells.cell_latitude))
    ).round(1)
    cells["cell_height_km_approx"] = round(111.32 * cell_degrees, 1)
    # Suppress cells with fewer than five sampled events in both scenarios.
    cells = cells.loc[(cells.reported_events >= min_samples) |
                      (cells.after_candidate_dedup >= min_samples)]
    cells = cells.sort_values(["reported_events", "cell_latitude", "cell_longitude"],
                              ascending=[False, True, True])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(output_dir / "spatial_hotspot_cells.csv", index=False)

    primary = _dbscan_summary(points, eps_km=eps_km, min_samples=min_samples)
    sensitivity = _dbscan_summary(deduped, eps_km=eps_km, min_samples=min_samples)
    summary: dict[str, Any] = {
        "analysis": "exploratory_spatial_concentration_of_media_reported_fatal_crashes",
        "input_rows": int(len(events)),
        "events_within_broad_india_envelope": int(len(points)),
        "coordinates_excluded_or_unparsed": int(len(events) - len(points)),
        "candidate_duplicate_rows_flagged": int(duplicate_group.sum()),
        "candidate_duplicate_signature_groups": duplicate_group_count,
        "candidate_duplicate_rows_retained_in_primary": True,
        "candidate_duplicate_rows_removed_only_in_sensitivity": int(duplicate_candidate.sum()),
        "cell_size_degrees": cell_degrees,
        "cell_minimum_events_for_export": min_samples,
        "coarse_cell_count_exported": int(len(cells)),
        "dbscan_primary": primary,
        "dbscan_after_candidate_duplicate_collapse": sensitivity,
        "limitations": [
            "All input rows are selected fatal crashes reported by a single newspaper; clusters are reported-event concentration, not crash incidence or road risk.",
            "The coordinates were parsed and checked only against a broad India envelope; positional accuracy and road-network matching were not audited.",
            "No traffic-volume, vehicle-kilometre, road-length, or population exposure denominator is available.",
            "Conservative candidate duplicate signatures are surfaced. The primary result retains them; a sensitivity run collapses signatures, which may also merge distinct events.",
            "DBSCAN depends on the chosen distance/minimum-support parameters. Clusters are exploratory review candidates, not official black spots.",
            "Coarse grid cells with fewer than five events are suppressed; output cell centroids are approximate and should not be reverse-geocoded to individuals.",
        ],
    }
    (output_dir / "spatial_hotspot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    top_cells = cells.head(10)
    table = "\n".join(
        f"| {row.cell_latitude:.3f} | {row.cell_longitude:.3f} | {row.reported_events} | "
        f"{row.after_candidate_dedup} |"
        for row in top_cells.itertuples(index=False)
    ) or "| No cells met the display support threshold | — | — | — |"
    report = f"""# Spatial concentration analysis: media-reported fatal crashes

## Purpose and interpretation

This analysis identifies coarse spatial concentrations in the Mendeley v7 structured sample of fatal crashes reported by *The Times of India*. It does **not** identify Indian road-crash risk, causal hazards, official black spots, or sites for automatic intervention. The source has no all-crash denominator, traffic exposure, positional-accuracy audit, or validated road-network linkage.

## Coverage and method

- Input rows: {len(events):,}; coordinates inside the broad India QA envelope: {len(points):,}; excluded/unparsed: {len(events) - len(points):,}.
- {int(duplicate_group.sum())} rows across {duplicate_group_count} conservative date/state/location/fatality/injury/vehicle signatures are flagged for review and retained in the primary analysis. The sensitivity analysis collapses {int(duplicate_candidate.sum())} repeated rows; a signature can merge distinct incidents.
- Coarse concentration grid: {cell_degrees:.2f}° latitude × {cell_degrees:.2f}° longitude (about {111.32 * cell_degrees:.1f} km north–south; east–west width varies by latitude). Cells with fewer than {min_samples} events are suppressed. Cell centroids are approximate.
- Distance clustering: haversine DBSCAN with eps={eps_km:.0f} km and min_samples={min_samples}; outputs are checked again after candidate-signature collapse.
- No road snapping, geocoding correction, causal attribution, risk normalization, or public low-count release is performed.

## Primary result

- {primary['cluster_count']} DBSCAN clusters contained {primary['clustered_events']:,} of {len(points):,} sampled events; {primary['noise_events']:,} fell outside those clusters under the selected parameters.
- Candidate-duplicate sensitivity: {sensitivity['cluster_count']} clusters and {sensitivity['clustered_events']:,} clustered events after collapsing {int(duplicate_candidate.sum())} repeated rows from {duplicate_group_count} flagged signatures.
- The output grid contains {len(cells):,} coarse cells meeting the five-event display threshold. See `data/processed/spatial_hotspot_cells.csv` for aggregate-only values.

### Highest-support coarse cells

| Approx. cell latitude | Approx. cell longitude | Reported sample events | After candidate-signature collapse |
|---:|---:|---:|---:|
{table}

This is a within-source concentration summary. A higher count can reflect population, traffic, reporting, sample construction, or duplicate issues. It is not evidence that a segment is comparatively dangerous.

## Cluster summaries

""" + "\n".join(
        f"- Cluster centered near **{row['center_latitude_rounded_0_1deg']:.1f}°, "
        f"{row['center_longitude_rounded_0_1deg']:.1f}°**: {row['events']:,} reports, "
        f"{row['states']} state labels, dates {row['date_min']} to {row['date_max']}."
        for row in primary["clusters"][:10]
    ) + "\n\n## Limitations and next steps\n\n" + "\n".join(
        f"- {item}" for item in summary["limitations"]
    ) + "\n\nFor a defensible engineering shortlist, obtain authorized representative crash records, audit coordinate quality, match events to a versioned road network, add matching-period vehicle-kilometres travelled or another exposure measure, and validate on held-out spatial blocks and future periods. Any candidate requires qualified field review.\n"
    report_path.write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cell-degrees", type=float, default=0.25)
    parser.add_argument("--eps-km", type=float, default=25.0)
    parser.add_argument("--min-samples", type=int, default=5)
    args = parser.parse_args()
    result = analyze_spatial(args.input, args.output_dir, args.report,
                             cell_degrees=args.cell_degrees, eps_km=args.eps_km,
                             min_samples=args.min_samples)
    print(json.dumps({
        "events_in_envelope": result["events_within_broad_india_envelope"],
        "cells_exported": result["coarse_cell_count_exported"],
        "clusters": result["dbscan_primary"]["cluster_count"],
        "clustered_events": result["dbscan_primary"]["clustered_events"],
        "report": str(args.report),
    }, indent=2))


if __name__ == "__main__":
    main()
