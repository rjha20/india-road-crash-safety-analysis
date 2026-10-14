# Spatial concentration analysis: media-reported fatal crashes

## Purpose and interpretation

This analysis identifies coarse spatial concentrations in the Mendeley v7 structured sample of fatal crashes reported by *The Times of India*. It does **not** identify Indian road-crash risk, causal hazards, official black spots, or sites for automatic intervention. The source has no all-crash denominator, traffic exposure, positional-accuracy audit, or validated road-network linkage.

## Coverage and method

- Input rows: 2,898; coordinates inside the broad India QA envelope: 2,898; excluded/unparsed: 0.
- 10 rows across 5 conservative date/state/location/fatality/injury/vehicle signatures are flagged for review and retained in the primary analysis. The sensitivity analysis collapses 5 repeated rows; a signature can merge distinct incidents.
- Coarse concentration grid: 0.25° latitude × 0.25° longitude (about 27.8 km north–south; east–west width varies by latitude). Cells with fewer than 5 events are suppressed. Cell centroids are approximate.
- Distance clustering: haversine DBSCAN with eps=25 km and min_samples=5; outputs are checked again after candidate-signature collapse.
- No road snapping, geocoding correction, causal attribution, risk normalization, or public low-count release is performed.

## Primary result

- 74 DBSCAN clusters contained 1,977 of 2,898 sampled events; 921 fell outside those clusters under the selected parameters.
- Candidate-duplicate sensitivity: 74 clusters and 1,976 clustered events after collapsing 5 repeated rows from 5 flagged signatures.
- The output grid contains 99 coarse cells meeting the five-event display threshold. See `data/processed/spatial_hotspot_cells.csv` for aggregate-only values.

### Highest-support coarse cells

| Approx. cell latitude | Approx. cell longitude | Reported sample events | After candidate-signature collapse |
|---:|---:|---:|---:|
| 22.625 | 88.375 | 74 | 74 |
| 22.625 | 75.875 | 50 | 50 |
| 28.625 | 77.125 | 49 | 49 |
| 12.875 | 77.625 | 42 | 42 |
| 18.625 | 73.875 | 42 | 42 |
| 19.875 | 73.875 | 38 | 38 |
| 13.125 | 80.125 | 34 | 34 |
| 19.125 | 72.875 | 34 | 34 |
| 28.625 | 77.375 | 33 | 33 |
| 20.125 | 73.875 | 32 | 32 |

This is a within-source concentration summary. A higher count can reflect population, traffic, reporting, sample construction, or duplicate issues. It is not evidence that a segment is comparatively dangerous.

## Cluster summaries

- Cluster centered near **28.6°, 77.2°**: 208 reports, 4 state labels, dates 2022-01-12 to 2023-12-26.
- Cluster centered near **18.7°, 73.7°**: 186 reports, 1 state labels, dates 2022-01-01 to 2023-12-27.
- Cluster centered near **30.7°, 76.7°**: 129 reports, 5 state labels, dates 2022-01-24 to 2023-11-27.
- Cluster centered near **22.7°, 75.9°**: 106 reports, 1 state labels, dates 2022-01-04 to 2023-12-26.
- Cluster centered near **13.0°, 80.1°**: 101 reports, 2 state labels, dates 2021-12-30 to 2023-11-13.
- Cluster centered near **20.0°, 73.8°**: 101 reports, 1 state labels, dates 2022-01-09 to 2023-12-17.
- Cluster centered near **13.0°, 77.6°**: 99 reports, 2 state labels, dates 2022-01-02 to 2023-12-20.
- Cluster centered near **22.6°, 88.4°**: 84 reports, 1 state labels, dates 2022-01-14 to 2023-12-22.
- Cluster centered near **23.3°, 77.4°**: 70 reports, 1 state labels, dates 2022-01-25 to 2023-12-26.
- Cluster centered near **17.7°, 83.2°**: 61 reports, 1 state labels, dates 2022-01-01 to 2023-12-23.

## Limitations and next steps

- All input rows are selected fatal crashes reported by a single newspaper; clusters are reported-event concentration, not crash incidence or road risk.
- The coordinates were parsed and checked only against a broad India envelope; positional accuracy and road-network matching were not audited.
- No traffic-volume, vehicle-kilometre, road-length, or population exposure denominator is available.
- Conservative candidate duplicate signatures are surfaced. The primary result retains them; a sensitivity run collapses signatures, which may also merge distinct events.
- DBSCAN depends on the chosen distance/minimum-support parameters. Clusters are exploratory review candidates, not official black spots.
- Coarse grid cells with fewer than five events are suppressed; output cell centroids are approximate and should not be reverse-geocoded to individuals.

For a defensible engineering shortlist, obtain authorized representative crash records, audit coordinate quality, match events to a versioned road network, add matching-period vehicle-kilometres travelled or another exposure measure, and validate on held-out spatial blocks and future periods. Any candidate requires qualified field review.
