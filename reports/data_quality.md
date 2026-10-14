# Crash data quality and coverage

## Source and scope

- **Source:** Ashutosh and Chand, *Media Reported Road Traffic Crash Data*, Mendeley Data v7, DOI [10.17632/bc5sv6wnd9.7](https://doi.org/10.17632/bc5sv6wnd9.7), CC BY 4.0 for the dataset. The structured workbook was downloaded from the v7 Data Files folder.
- **Input file:** `news_crashes.xlsx`; SHA-256 `02ece51f1c04ea44fe3ccbafac150e4c937be79b1d5c1216dc68ab32c66a4228`.
- **Observed table:** 2,898 records and 17 physical columns (serial number plus 16 dictionary variables).
- **Population represented:** fatal crashes reported by The Times of India, not all Indian crashes. The data is editorially selected and has no traffic-exposure denominator; event totals and mapped concentrations are not comparative road-risk estimates.
- **Narratives:** this structured workbook contains no article title or body. Raw narrative analysis is restricted to the separate 2022 article file; 2023 raw-file access returned an error and is excluded.

## Completeness

Missingness uses empty/NA-like tokens. The source value `Nil` for Million Plus City is retained as “Not listed as a million-plus city,” not interpreted as an unknown value.

| Source field | Missing | Missing % |
|---|---:|---:|
| `Age` | 729 | 25.16% |
| `Gender` | 263 | 9.08% |
| `Vehicle/Object 2` | 0 | 0.00% |
| `Million Plus City` | 0 | 0.00% |
| `Article Date` | 0 | 0.00% |
| `Road Type` | 0 | 0.00% |
| `Vehicle 1` | 0 | 0.00% |
| `Location` | 0 | 0.00% |
| `Injured` | 0 | 0.00% |
| `LatLong` | 0 | 0.00% |
| `Crash Type` | 0 | 0.00% |
| `State` | 0 | 0.00% |
| `Killed` | 0 | 0.00% |
| `Crash Date` | 0 | 0.00% |
| `Crash Day` | 0 | 0.00% |
| `Month` | 0 | 0.00% |
| `S. No.` | 0 | 0.00% |

`Age` and `Gender` have source missingness, and may include multi-victim values. Their source values were deliberately omitted from processed event rows because they are not needed for this project’s model.

## Integrity and geocoding checks

- Source serial number unique: **True**; missing serials: 0.
- Exact duplicate rows including the serial: 0.
- Rows flagged by a conservative duplicate signature (date, state, location, fatalities, injuries, and both vehicle fields): 10 across 5 signature group(s). These rows are surfaced in `data/processed/crash_duplicate_signatures.csv`, not silently removed. The signature can flag distinct events sharing a place/date, so it requires review.
- Crash-date coverage: 2021-12-30 through 2023-12-30; article-date coverage: 2022-01-01 through 2023-12-31.
- Source month disagrees with parsed crash-date month on 164 rows; weekday disagrees on 2 rows. Dates were not “corrected” from those secondary text fields.
- Coordinates parsed: 2,898/2,898; within a broad India envelope: 2,898; outside envelope: 0; unparsed: 0. This range check is not a positional accuracy audit or road snap.
- States represented: 30.

## Outcome counts and cautions

- Fatalities recorded in the selected sample: 6,584; injuries: 7,812.
- Rows with one recorded death: 1,442; rows with two or more: 1,456.
- Every structured row belongs to a **fatal-crash sample**. Modeling the number of deaths is therefore a conditional description among reported fatal crashes, not prediction of whether a road will have a crash, not national crash risk, and not a causal explanation.
- Coordinates/location labels were extracted from reports. The envelope check cannot establish geocoding precision; a road-network join still requires manual verification and match-confidence reporting.

## Reproduction and output handling

Run `python -m road_safety.crash_data` from the project environment. The row-level `crash_events_clean.csv` remains local and gitignored. The shareable `reported_event_grid_public.csv` contains only year/state/coarse-cell counts and fatalities; cells with fewer than 5 sampled records per year/state slice are suppressed. It omits exact coordinates, dates, location descriptions, article text, and source row identifiers. Other processed CSV/JSON outputs are generated under `data/processed/`. The pipeline omits victim Age/Gender values and all raw news article text from its processed files.
