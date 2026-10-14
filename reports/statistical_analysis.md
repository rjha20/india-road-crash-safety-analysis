# Statistical baseline: India road accidents, 2019–2023

**Source:** Ministry of Road Transport and Highways (MoRTH), *Road Accidents in India 2023*.  
**Analysis date:** 7 October 2026.  
**Decision supported:** Set national/state context and identify where a deeper, geocoded engineering review could be worthwhile. This is not a road-segment hotspot ranking.

## Executive readout

- MoRTH reports **480,583 road crashes and 172,890 fatalities in 2023**, up **4.18%** and **2.61%**, respectively, from 2022. The arithmetic increase is 19,271 reported crashes and 4,399 deaths.
- Tamil Nadu had the highest reported all-road crash count (**67,213**); Uttar Pradesh had the highest reported fatality count (**23,652**). The report’s top ten states together account for **78.60%** of crashes and **73.90%** of fatalities.
- Among national highways, MoRTH reports **150,177 crashes** and **63,112 fatalities** in 2023. The report’s top ten states account for 114,601 NH crashes (76.31%) and 46,690 NH fatalities (73.98%). NH state tables list only their respective top ten; missing CSV values mean “not listed in this table,” not zero.
- The all-road fatality-to-reported-crash ratio is **36.0 fatalities per 100 reported crashes** nationally. It is a count ratio that can exceed 100 and is not a probability or an exposure-adjusted rate.
- This source contains state/UT totals, not crash coordinates, road-segment identifiers, or traffic volume. It cannot identify crash hotspots or compare road risk. Use it for aggregate context and source reconciliation; site-level analysis needs geocoded records plus exposure such as vehicle-kilometres travelled (VKT).

## Source and extracted data

Primary source: [MoRTH, *Road Accidents in India 2023* (official PDF)](https://morth.gov.in/backend/documents/uploaded/Road-Accident-in-India-2023-Publications.pdf).

The source PDF is retained at `data/raw/sources/Road-Accident-in-India-2023-Publications.pdf` (SHA-256: `901505420D5B387052AC51E36B4BDCE0274F6326B614F9EBCEE6E7954D45B5B0`). The processed state/UT table is `data/processed/morth_nh_2023_state.csv`; it has 37 source rows, including historical administrative entries, and preserves blanks as missing values.

| PDF table | Printed pages | Data represented in the CSV |
|---|---:|---|
| Table 5.1 | 66 | State/UT all-road crash counts, 2019–2023 |
| Table 5.5 | 69 | NH crash counts, 2019–2023, only for the report’s top ten states |
| Table 5.6 | 71 | State/UT road-fatality counts, 2019–2023 |
| Table 5.10 | 74 | NH fatality counts, 2019–2023, only for the report’s top ten states |

Table 5.5 and Table 5.10 select top-ten lists independently. Therefore the CSV contains NH crash values for ten states and NH fatality values for another ten-state list; only nine states appear in both. Do not interpret a blank NH value as a zero or calculate complete cross-state comparisons from these truncated lists. The report’s national totals are stored in the analysis module as published totals, separately from the state rows.

The source report notes boundary changes: Jammu & Kashmir includes Ladakh for 2019–2020; its Dadra & Nagar Haveli footnote says that the series includes Daman & Diu data. Table 5.1 and Table 5.6 have different missing-value patterns for Daman & Diu. The extraction preserves the displayed table values and blanks rather than guessing or harmonizing them. The paired 2022–2023 state/UT summaries therefore use 36 units with both years reported and omit the separate Daman & Diu row for that comparison.

## Descriptive findings

### National trend and state totals

| Measure | 2022 | 2023 | Change | Change % |
|---|---:|---:|---:|---:|
| All-road reported crashes | 461,312 | 480,583 | +19,271 | +4.18% |
| All-road reported fatalities | 168,491 | 172,890 | +4,399 | +2.61% |
| NH reported crashes | 151,997 | 150,177 | −1,820 | −1.20% |
| NH reported fatalities | 61,038 | 63,112 | +2,074 | +3.40% |

These are administrative counts supplied to MoRTH. The increase in NH deaths alongside fewer NH crashes is a reason to examine crash severity, reporting definitions, vehicle mix, and exposure with better data; these aggregates do not identify a cause.

| 2023 all-road count rank | Reported crashes |  | Reported fatalities |  |
|---:|---|---:|---|---:|
| 1 | Tamil Nadu | 67,213 | Uttar Pradesh | 23,652 |
| 2 | Madhya Pradesh | 55,327 | Tamil Nadu | 18,347 |
| 3 | Kerala | 48,091 | Maharashtra | 15,366 |
| 4 | Uttar Pradesh | 44,534 | Madhya Pradesh | 13,798 |
| 5 | Karnataka | 43,440 | Karnataka | 12,321 |

The ten highest-count state/UTs retain the same membership between 2022 and 2023 for both all-road crashes and fatalities. Across the 36 comparable units, Spearman rank correlation is **0.998** for crash counts (unit-bootstrap 95% interval **0.989–1.000**) and **0.993** for fatality counts (**0.967–0.999**). This indicates stable ordering of large administrative totals over this one-year comparison. It does not establish stable road-segment risk or generalization to another period.

The 2023 cross-state association between reported crash and fatality counts is Spearman **ρ = 0.938** (administrative-unit bootstrap 95% interval **0.843–0.979**). Large states tend to have more of both outcomes in absolute counts; population, fleet size, road length, travel and reporting coverage are not controlled, so this is not evidence that one count causes the other.

### Paired change across state/UT rows, 2022 to 2023

The following intervals resample the 36 state/UT rows with replacement. They describe variation among those units; they are not confidence intervals for MoRTH’s exact national totals.

| Metric | Units higher | Lower | Median signed count change (bootstrap 95% interval) | Median state percentage change (bootstrap 95% interval) | Exact two-sided sign-test p |
|---|---:|---:|---:|---:|---:|
| All-road crashes | 24 | 12 | +118 (+0 to +206) | +1.53% (−2.55% to +3.22%) | 0.065 |
| Fatalities | 22 | 14 | +16 (−3 to +88) | +1.24% (−0.73% to +3.87%) | 0.243 |

A sign-flip permutation test of the **mean signed unit-level count change** returned nominal two-sided p-values of 0.0047 for crashes and 0.0034 for fatalities. The exact sign test instead evaluates only the balance of units increasing versus decreasing. These answer different questions, and the permutation result is sensitive to the magnitude of large-state changes. Because this is the set of reporting jurisdictions rather than a randomized or sampled set of crashes, these p-values are exploratory diagnostics, not evidence of a causal effect or uncertainty in the national trend. The published totals give the national year-over-year changes directly.

### Fatalities per reported crash: a descriptive severity proxy only

Dividing fatalities by crashes gives **36.0 per 100 reported crashes nationally** and a median **40.7 per 100** across the 36 state/UT rows. Ratios vary widely: examples among the highest are Mizoram (90.6), Bihar (80.6), and Jharkhand (78.5); among the lowest are Kerala (8.5), Goa (10.2), and Puducherry (11.4). Lakshadweep’s 0/1 yields 0.0 and illustrates how tiny denominators make such ratios unstable.

Several people may die in one crash, and the number of crashes is not the number of vehicles or people exposed. These ratios are especially sensitive to small counts, definitions, and completeness. **Do not label them fatality probability, fatality rate, or state road-safety risk.**

### National Highway context

The source lists the 2023 NH fatality top five as Uttar Pradesh (8,446), Tamil Nadu (6,258), Maharashtra (5,780), Madhya Pradesh (4,476), and Karnataka (4,383). NH crash-count leaders in Table 5.5 are Tamil Nadu (20,582), Uttar Pradesh (15,184), Madhya Pradesh (14,561), Karnataka (14,270), and Maharashtra (10,881). The two lists differ in order and membership, and their denominators are not vehicle-kilometres or traffic volumes.

Nationally, the NH fatality-to-crash ratio is 42.0 per 100, compared with 36.0 per 100 on all roads. This descriptive quotient does not adjust for traffic exposure, road function, case mix, or reporting and cannot support a claim that NH travel is riskier.

## Statistical methods implemented

`src/road_safety/statistics.py` uses only the Python standard library. It provides:

1. Strict CSV loading, unique jurisdiction checks, and preservation of blank/NA values as missing.
2. Paired 2022–2023 change summaries: signed counts, per-unit percentage changes, increase/decrease counts, bootstrap percentile intervals for means and medians, an exact sign test, and a seeded Monte Carlo sign-flip test for the mean.
3. Spearman rank association with mid-ranks for ties, paired-year rank stability, top-ten overlap, and state-unit bootstrap intervals.
4. Descriptive fatality-to-crash quotients and the coverage/share calculation for the two separate NH top-ten tables.
5. Explicit interpretive notes separating administrative count patterns from exposure-adjusted risk and crash-level uncertainty.

The bootstrap samples jurisdictions, not incidents. The sign-flip test assumes sign exchangeability under its null; neither procedure compensates for source under-reporting, boundary changes, incomplete data, or lack of exposure. Do not use the reported state rankings to allocate road engineering resources without segment-level corroboration.

## Reproduction

From the project root, with Python 3.11 or newer (the project environment is locked with `uv`):

```powershell
python src/road_safety/statistics.py --reps 10000
```

The script prints a deterministic JSON summary (fixed seed) and reads `data/processed/morth_nh_2023_state.csv` by default. Supply `--data PATH` to analyze another compatible CSV. No third-party packages are required for this module.

## What this baseline cannot answer

- Which specific road, junction, or kilometre is a hotspot.
- Whether one state, road type, or segment has greater crash risk after accounting for traffic volume, length, and population.
- Whether a change reflects road safety, travel volume, crash severity, police reporting, data revisions, or administrative boundaries.
- Whether a photographed defect contributed to a crash.

For those questions, join deduplicated geocoded crash records to a versioned road network and matching time-period traffic exposure (preferably VKT), then evaluate segment counts/rates using spatially held-out and future-period validation. Until those inputs are available, use this MoRTH analysis as a sourced administrative baseline only.
