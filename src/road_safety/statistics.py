"""Descriptive/statistical baseline for official MoRTH state and NH aggregates.

This module intentionally does not estimate segment crash risk: the input is
administrative aggregates and contains no traffic exposure denominator.
Only Python's standard library is required.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

YEARS = range(2019, 2024)
DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / "processed" / "morth_nh_2023_state.csv"
NATIONAL_ROAD_TOTALS = {
    "accidents": {2022: 461_312, 2023: 480_583},
    "fatalities": {2022: 168_491, 2023: 172_890},
}
NATIONAL_NH_TOTALS = {
    "accidents": {2022: 151_997, 2023: 150_177},
    "fatalities": {2022: 61_038, 2023: 63_112},
}


def _number(value: str | None) -> int | None:
    if value is None or not value.strip() or value.strip().upper() in {"NA", "N/A", "#N/A"}:
        return None
    return int(value.replace(",", "").strip())


def read_state_table(path: Path = DEFAULT_DATA) -> list[dict[str, Any]]:
    """Read state/UT rows; blanks remain missing, never silently become zero."""
    required = {"state_ut"}
    for year in YEARS:
        required.update({f"accidents_{year}", f"fatalities_{year}", f"nh_accidents_{year}"})
    required.add("nh_fatalities_2023")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"Missing required columns in {path}: {missing}")
        rows: list[dict[str, Any]] = []
        for source_row in reader:
            state = (source_row.get("state_ut") or "").strip()
            if not state or state.casefold() in {"total", "all india", "india"}:
                continue
            row: dict[str, Any] = {"state_ut": state}
            for year in YEARS:
                for metric in ("accidents", "fatalities", "nh_accidents"):
                    row[f"{metric}_{year}"] = _number(source_row.get(f"{metric}_{year}"))
            row["nh_fatalities_2023"] = _number(source_row.get("nh_fatalities_2023"))
            rows.append(row)
    names = [row["state_ut"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("state_ut must be unique; resolve duplicate or changed boundaries first")
    if not rows:
        raise ValueError(f"No state/UT records found in {path}")
    return rows


def _pairs(rows: Sequence[dict[str, Any]], metric: str, start: int, end: int) -> list[tuple[str, int, int]]:
    return [
        (row["state_ut"], row[f"{metric}_{start}"], row[f"{metric}_{end}"])
        for row in rows
        if row[f"{metric}_{start}"] is not None and row[f"{metric}_{end}"] is not None
    ]


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _bootstrap_interval(
    values: Sequence[float], statistic: str, *, reps: int, seed: int
) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    estimates: list[float] = []
    n = len(values)
    for _ in range(reps):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        if statistic == "mean":
            estimates.append(sum(sample) / n)
        elif statistic == "median":
            estimates.append(_percentile(sample, 0.5))
        else:
            raise ValueError(f"Unsupported bootstrap statistic: {statistic}")
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def _sign_flip_p_value(differences: Sequence[float], *, reps: int, seed: int) -> float | None:
    """Monte Carlo two-sided paired sign-flip test for mean change."""
    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return 1.0 if differences else None
    observed = abs(sum(differences) / len(differences))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(reps):
        randomized_mean = abs(sum(value if rng.getrandbits(1) else -value for value in differences) / len(differences))
        if randomized_mean >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (reps + 1)


def _sign_test_p_value(differences: Sequence[float]) -> float | None:
    """Exact two-sided binomial sign test, with ties omitted."""
    signs = [value for value in differences if value != 0]
    n = len(signs)
    if n == 0:
        return 1.0 if differences else None
    k = min(sum(value > 0 for value in signs), sum(value < 0 for value in signs))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def paired_change_summary(
    rows: Sequence[dict[str, Any]], metric: str, start: int = 2022, end: int = 2023,
    *, reps: int = 10_000, seed: int = 42,
) -> dict[str, Any]:
    paired = _pairs(rows, metric, start, end)
    differences = [end_value - start_value for _, start_value, end_value in paired]
    pct_changes = [100 * (end_value - start_value) / start_value for _, start_value, end_value in paired if start_value > 0]
    mean_change = sum(differences) / len(differences) if differences else None
    median_change = _percentile(differences, 0.5) if differences else None
    mean_pct = sum(pct_changes) / len(pct_changes) if pct_changes else None
    median_pct = _percentile(pct_changes, 0.5) if pct_changes else None
    return {
        "metric": metric,
        "period": f"{start}-{end}",
        "paired_units": len(paired),
        "increased": sum(value > 0 for value in differences),
        "unchanged": sum(value == 0 for value in differences),
        "decreased": sum(value < 0 for value in differences),
        "mean_signed_count_change": mean_change,
        "mean_signed_change_bootstrap_95pct_interval": _bootstrap_interval(differences, "mean", reps=reps, seed=seed),
        "median_signed_count_change": median_change,
        "median_signed_change_bootstrap_95pct_interval": _bootstrap_interval(differences, "median", reps=reps, seed=seed + 1),
        "median_state_pct_change": median_pct,
        "median_state_pct_change_bootstrap_95pct_interval": _bootstrap_interval(pct_changes, "median", reps=reps, seed=seed + 2),
        "mean_state_pct_change": mean_pct,
        "paired_sign_flip_two_sided_p": _sign_flip_p_value(differences, reps=reps, seed=seed + 3),
        "exact_sign_test_two_sided_p": _sign_test_p_value(differences),
        "unit_changes": [
            {"state_ut": name, "start": before, "end": after,
             "absolute_change": after - before,
             "percent_change": (100 * (after - before) / before) if before > 0 else None}
            for name, before, after in paired
        ],
        "interpretation_note": "Resampling/permutation is across administrative units. It is not crash-level measurement uncertainty or a national rate test.",
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = ((i + 1) + j) / 2
        for position in range(i, j):
            ranks[indexed[position][0]] = rank
        i = j
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    xx = sum((v - mean_x) ** 2 for v in x)
    yy = sum((v - mean_y) ** 2 for v in y)
    if xx == 0 or yy == 0:
        return None
    return sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / math.sqrt(xx * yy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    return _pearson(_average_ranks(x), _average_ranks(y))


def _bootstrap_spearman_interval(x: Sequence[float], y: Sequence[float], *, reps: int, seed: int) -> list[float | None]:
    if len(x) < 3:
        return [None, None]
    rng = random.Random(seed)
    estimates: list[float] = []
    n = len(x)
    for _ in range(reps):
        indexes = [rng.randrange(n) for _ in range(n)]
        value = spearman([x[i] for i in indexes], [y[i] for i in indexes])
        if value is not None:
            estimates.append(value)
    if not estimates:
        return [None, None]
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def _ranked_names(rows: Sequence[dict[str, Any]], column: str) -> list[str]:
    valid = [row for row in rows if row[column] is not None]
    return [row["state_ut"] for row in sorted(valid, key=lambda row: (-row[column], row["state_ut"]))]


def rank_stability(rows: Sequence[dict[str, Any]], metric: str, start: int = 2022, end: int = 2023, *, reps: int = 10_000, seed: int = 2023, top_n: int = 10) -> dict[str, Any]:
    paired = _pairs(rows, metric, start, end)
    before = [item[1] for item in paired]
    after = [item[2] for item in paired]
    rank_corr = spearman(before, after)
    before_names = [name for name, _, _ in sorted(paired, key=lambda item: (-item[1], item[0]))]
    after_names = [name for name, _, _ in sorted(paired, key=lambda item: (-item[2], item[0]))]
    before_top = before_names[:top_n]
    after_top = after_names[:top_n]
    overlap = sorted(set(before_top).intersection(after_top))
    return {
        "metric": metric,
        "period": f"{start}-{end}",
        "paired_units": len(paired),
        "spearman_rank_correlation": rank_corr,
        "rank_correlation_bootstrap_95pct_interval": _bootstrap_spearman_interval(before, after, reps=reps, seed=seed),
        "top_n": top_n,
        "top_n_overlap_count": len(overlap),
        "top_n_overlap": overlap,
        "entered_top_n": sorted(set(after_top).difference(before_top)),
        "left_top_n": sorted(set(before_top).difference(after_top)),
        "top_n_2022": before_top,
        "top_n_2023": after_top,
        "interpretation_note": "Ranks describe reported aggregate counts, not exposure-adjusted risk. The bootstrap interval resamples state/UT units.",
    }


def summarize(rows: Sequence[dict[str, Any]], *, reps: int = 10_000) -> dict[str, Any]:
    active = [row for row in rows if row["accidents_2023"] is not None and row["fatalities_2023"] is not None]
    count_vs_deaths_rho = spearman(
        [row["accidents_2023"] for row in active],
        [row["fatalities_2023"] for row in active],
    )
    severity_proxy = [
        (row["state_ut"], 100 * row["fatalities_2023"] / row["accidents_2023"])
        for row in active if row["accidents_2023"] > 0
    ]
    severity_proxy.sort(key=lambda pair: pair[1], reverse=True)
    nh_rows = [row for row in rows if row["nh_accidents_2023"] is not None and row["nh_fatalities_2023"] is not None]
    nh_rank = sorted(nh_rows, key=lambda row: (-row["nh_fatalities_2023"], row["state_ut"]))
    national_changes = {
        metric: {
            "2022": totals[2022], "2023": totals[2023],
            "absolute_change": totals[2023] - totals[2022],
            "percent_change": 100 * (totals[2023] - totals[2022]) / totals[2022],
        }
        for metric, totals in NATIONAL_ROAD_TOTALS.items()
    }
    nh_national_changes = {
        metric: {
            "2022": totals[2022], "2023": totals[2023],
            "absolute_change": totals[2023] - totals[2022],
            "percent_change": 100 * (totals[2023] - totals[2022]) / totals[2022],
        }
        for metric, totals in NATIONAL_NH_TOTALS.items()
    }
    nh_top10_sum = sum(row["nh_accidents_2023"] for row in rows if row["nh_accidents_2023"] is not None)
    nh_fatal_top10_sum = sum(row["nh_fatalities_2023"] for row in rows if row["nh_fatalities_2023"] is not None)
    return {
        "source": "MoRTH, Road Accidents in India 2023, Tables 5.1, 5.5, 5.6 and 5.10",
        "source_url": "https://morth.gov.in/backend/documents/uploaded/Road-Accident-in-India-2023-Publications.pdf",
        "input_rows": len(rows),
        "comparable_2023_state_ut_units": len(active),
        "2023_top_five_by_reported_accidents": [
            {"state_ut": row["state_ut"], "accidents": row["accidents_2023"]}
            for row in sorted(active, key=lambda item: (-item["accidents_2023"], item["state_ut"]))[:5]
        ],
        "2023_top_five_by_reported_fatalities": [
            {"state_ut": row["state_ut"], "fatalities": row["fatalities_2023"]}
            for row in sorted(active, key=lambda item: (-item["fatalities_2023"], item["state_ut"]))[:5]
        ],
        "2023_top_ten_share_of_all_india_reported_counts_pct": {
            "accidents": 100 * sum(row["accidents_2023"] for row in sorted(active, key=lambda item: (-item["accidents_2023"], item["state_ut"]))[:10]) / NATIONAL_ROAD_TOTALS["accidents"][2023],
            "fatalities": 100 * sum(row["fatalities_2023"] for row in sorted(active, key=lambda item: (-item["fatalities_2023"], item["state_ut"]))[:10]) / NATIONAL_ROAD_TOTALS["fatalities"][2023],
        },
        "official_all_india_reported_change": national_changes,
        "official_all_india_NH_change": nh_national_changes,
        "state_count_spearman_accidents_vs_fatalities_2023": count_vs_deaths_rho,
        "state_count_spearman_bootstrap_95pct_interval": _bootstrap_spearman_interval(
            [row["accidents_2023"] for row in active], [row["fatalities_2023"] for row in active], reps=reps, seed=51
        ),
        "reported_fatalities_per_100_reported_crashes_2023": {
            "all_india": 100 * NATIONAL_ROAD_TOTALS["fatalities"][2023] / NATIONAL_ROAD_TOTALS["accidents"][2023],
            "state_ut_median": _percentile([value for _, value in severity_proxy], 0.5),
            "highest_five": [{"state_ut": name, "per_100": value} for name, value in severity_proxy[:5]],
            "lowest_five": [{"state_ut": name, "per_100": value} for name, value in severity_proxy[-5:]],
            "label": "fatality-to-reported-crash ratio; not exposure-adjusted risk or probability of death",
        },
        "change_summary_accidents": paired_change_summary(rows, "accidents", reps=reps, seed=10),
        "change_summary_fatalities": paired_change_summary(rows, "fatalities", reps=reps, seed=20),
        "rank_stability_accidents": rank_stability(rows, "accidents", reps=reps, seed=30),
        "rank_stability_fatalities": rank_stability(rows, "fatalities", reps=reps, seed=40),
        "NH_2023_top10_coverage": {
            "states_with_both_NH_accidents_and_fatalities": len(nh_rows),
            "NH_accidents_top10_sum": nh_top10_sum,
            "NH_accidents_all_india": NATIONAL_NH_TOTALS["accidents"][2023],
            "NH_accident_share_top10_pct": 100 * nh_top10_sum / NATIONAL_NH_TOTALS["accidents"][2023],
            "NH_fatalities_top10_sum": nh_fatal_top10_sum,
            "NH_fatalities_all_india": NATIONAL_NH_TOTALS["fatalities"][2023],
            "NH_fatality_share_top10_pct": 100 * nh_fatal_top10_sum / NATIONAL_NH_TOTALS["fatalities"][2023],
            "NH_top_five_by_fatalities": [
                {"state_ut": row["state_ut"], "fatalities": row["nh_fatalities_2023"], "accidents": row["nh_accidents_2023"]}
                for row in nh_rank[:5]
            ],
            "NH_fatalities_per_100_crashes_all_india": 100 * NATIONAL_NH_TOTALS["fatalities"][2023] / NATIONAL_NH_TOTALS["accidents"][2023],
            "caveat": "State NH rows are the report's top-ten-state tables, not a complete state panel; no traffic exposure denominator is provided.",
        },
        "limitations": [
            "State/UT totals are aggregate reported events, not geocoded road-segment counts.",
            "Population, road length, and traffic exposure/VKT are absent; raw counts cannot rank road risk.",
            "Reporting definitions and administrative boundaries differ over time; Jammu & Kashmir includes Ladakh through 2020, and Dadra & Nagar Haveli/Daman & Diu footnotes require caution.",
            "Bootstrap intervals and sign/permutation tests resample administrative units and describe cross-unit change patterns; they do not quantify uncertainty in the official national totals.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the processed MoRTH state/UT CSV")
    parser.add_argument("--reps", type=int, default=10_000, help="Bootstrap/permutation repetitions (default: 10000)")
    args = parser.parse_args()
    if args.reps < 100:
        parser.error("--reps must be at least 100 for a useful Monte Carlo summary")
    result = summarize(read_state_table(args.data), reps=args.reps)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
