# India road-crash hotspot and hazard analysis

## Executive summary

This portfolio project joins several complementary analyses while keeping their source populations separate: official MoRTH state aggregates, 2,898 fatal crashes reported by *The Times of India* and compiled in Mendeley v7, 1,565 unique 2022 news articles, and an independent RDD2022 India road-damage image benchmark. It demonstrates statistical inference, spatial clustering, NLP, supervised ML, computer vision, data quality, and a local review dashboard.

It does **not** estimate Indian road-segment crash risk. The crash records are fatal-only and media-selected; there is no exposure denominator, no verified road-network match, and no representative all-crash event set. The imagery is not linked to the crash reports. Every location pattern here is an exploratory reported-event concentration that would need independent data and qualified engineering review.

## Decision question and scope

**Question:** What patterns can be reproducibly described in available public sources, and what additional evidence would be required before an engineer could assess candidate sites?

**Geography and period:** India; the Mendeley structured sample spans crash dates 30 December 2021–30 December 2023, article NLP is limited to calendar year 2022, and the official MoRTH baseline compares administrative counts through 2023. RDD2022 is a separate labeled road-image benchmark.

**Intended reader:** data-science reviewers, transport researchers, and road-safety engineers evaluating evidence and data requirements.

**Non-goals:** crash alerts, road-danger scores, official black-spot designation, driver attribution, causal conclusions, enforcement, or allocation of engineering funds.

## Sources and analytical separation

| Evidence layer | Records used | What it supports | What it does not support |
|---|---|---|---|
| MoRTH *Road Accidents in India 2023* | Published national, state/UT, and selected NH aggregates | Administrative counts and descriptive year-over-year context | Segment hotspots, traffic-adjusted risk, or complete cross-state NH comparisons; two NH state tables report different top-ten lists |
| Mendeley v7 structured compilation | 2,898 fatal crashes reported by one newspaper, with dates, attributes, and parsed coordinates | QA, coarse spatial concentrations, and conditional severity modeling within the sample | All-crash incidence, population risk, causality, or representative state comparisons |
| Mendeley v7 raw narrative file | 1,565 unique 2022 article bodies after exact-body de-duplication | Weak corpus-level keyword-mention summaries | Validated cause extraction, 2023 narrative trends, or NLP accuracy without human labels |
| RDD2022 India image subset | Road images with VOC defect boxes; analyzed independently | Visual defect classification/detection performance on a grouped benchmark split | A crash record, crash location, defect-to-crash link, or maintenance priority |

Source versions, retrieval status, rights notes, and hashes are in `data/raw/source_manifest.json`. Raw source material remains local and ignored by Git. In particular, the 2023 Mendeley raw-text endpoint returned an error payload and is excluded from analysis; the news prose is not redistributed in outputs.

## Data quality and preparation

The structured workbook has 2,898 rows across 30 state labels. The pipeline parsed all 2,898 coordinate pairs and all fell within a broad India bounding envelope. This is a range check only: coordinates were not independently geocoded, snapped to roads, or accuracy-audited. There were 164 disagreements between the source month field and parsed crash date, and two weekday mismatches; the date field was retained without silent correction.

Ten rows across five conservative same-date/state/location/outcome/vehicle signatures were flagged for review. The primary spatial and ML analyses retain all records; spatial sensitivity analysis removes five repeated rows (one per signature after the first). The signature can merge real separate events, so the sensitivity is not treated as definitive deduplication. Victim age/gender values and article text are omitted from processed row-level outputs.

See [data quality report](data_quality.md) for field-level missingness and [data policy](../docs/data_policy.md) for the release boundary.

## Findings

### Official administrative baseline

MoRTH reported 480,583 all-road crashes and 172,890 fatalities in 2023, increases of 4.18% and 2.61% from 2022. Tamil Nadu had the largest all-road crash count (67,213), while Uttar Pradesh had the largest fatality count (23,652). On National Highways, the report lists 150,177 crashes and 63,112 fatalities. These are administrative totals and do not contain a segment identifier or traffic-volume denominator.

The project computes paired state/UT changes, rank stability, bootstrap intervals, a sign test, and a sign-flip diagnostic. These procedures describe the published administrative units; they do not change the fact that the national totals themselves are reported counts. The quotient fatalities per 100 reported crashes is explicitly not a probability or exposure-adjusted rate.

### Spatial concentrations in the reported fatal-event sample

All 2,898 rows passed the broad coordinate envelope. With DBSCAN configured at 25 km and a minimum of five events, 74 clusters contained 1,977 sampled events; 921 were outside clusters at those settings. A quarter-degree grid produced 99 cells with at least five events. The largest cells by sample count were centered approximately near Kolkata (74), Indore (50), Delhi (49), Bengaluru (42), and Pune (42), based on cell centroids; other high-count cells were in the Pune/Mumbai region and Chennai. These are reporting-sample concentrations and should not be interpreted as comparative road safety.

After collapsing the five repeated rows from the five flagged duplicate signatures, DBSCAN still returned 74 clusters and contained 1,976 events. This small numerical change shows that the broad count is not driven only by the flagged rows, but says nothing about editorial selection or geocoding accuracy. DBSCAN results depend on the selected radius and minimum support; alternative parameters are a required sensitivity analysis before any research shortlist.

### Narrative NLP: weak mention counts only

The 2022 article file contained 1,568 source rows; three exact duplicate article bodies were removed for aggregate counting, leaving 1,565 unique articles. The most frequent weak dictionary matches were heavy-vehicle terms (829 articles; 53.0%), two-wheeler terms (604; 38.6%), and speed/unsafe-manoeuvre terms (600; 38.3%). Themes overlap and can miss language or match negation, chronology, attribution, or an article's framing. No manually reviewed gold set exists, so the project reports no precision, recall, F1, or validated cause labels.

These values describe a single newspaper's 2022 fatal-crash story corpus. They are not the fraction of all crashes caused by a factor.

### Structured ML: conditional fatality band

The supervised demonstration predicts whether an included event records at least two deaths versus one death, **conditional on already being in this media-reported fatal-crash sample**. Predictors are road type, crash type, and vehicle categories. State is used only for grouped validation; post-event injury counts, dates, coordinates, location, and victim fields are excluded.

Five-fold stratified grouped cross-validation held out state groups. The logistic model achieved 0.684 accuracy, 0.684 balanced accuracy, 0.675 F1, 0.746 ROC-AUC, and 0.742 average precision. A majority baseline achieved 0.455 accuracy and 0.370 F1. Fold ROC-AUC ranged from 0.707 to 0.809, and grouped bootstrap intervals are reported in the metrics artifact. This demonstrates a reproducible conditional classification workflow; it is neither road-risk prediction nor causal evidence and has no external or future-year validation.

### Computer vision: separate benchmark

The road-damage detector uses the RDD2022 India image set independently from the crash records. Its images have no demonstrated event-level linkage to these crashes. The archive contains 7,706 India training images with matching XMLs. Before filtering, the four target classes had 1,555 D00, 68 D10, 2,021 D20, and 3,187 D40 instances. Because RDD also labels other damage types, 1,056 images containing out-of-scope/malformed labels are excluded; 6,650 remain eligible, including 3,921 verified empty-annotation images. The eligible class counts are D00 1,419, D10 63, D20 1,635, and D40 2,956. Empty-target images are withheld from fitting after an observed numerical-loss failure but retained in validation, where they help measure false positives.

The implementation fine-tuned a COCO-pretrained Faster R-CNN MobileNetV3-Large for one epoch on 1,200 selected training images and evaluated 300 held-out images on CPU. On this pilot holdout, mAP50 was 0.126 and mAP50:95 was 0.039. At a fixed 0.5 confidence/IoU threshold, micro precision was 0.581, recall 0.092, and F1 0.159. Per-class AP50 was D00 0.045, D10 0.000, D20 0.389, and D40 0.072; D20 was the only class with nonzero recall at that fixed threshold. The held-out groups are blocks of sequential image IDs, a leakage-reduction proxy rather than verified geographic groups. D10 is especially sparse (63 eligible instances and only three in this validation run). These results show a working detection/evaluation pipeline, but the detector is underfit and not suitable for field use. The [vision evaluation report](vision_evaluation.md) documents class metrics, exact split counts, and checkpoint details; the dashboard summarizes them.

## Research interpretation

The components answer different descriptive questions and must not be combined into a synthetic hazard score. A high-density coarse cell from the newspaper sample, a MoRTH state total, a keyword mention, and an RDD image detection are not commensurate evidence. The project therefore keeps these views separate in both the code and the dashboard.

The most defensible conclusion is about data readiness: public aggregate and media sources allow source-audited descriptive demonstrations, but the available evidence does not identify road segments with elevated exposure-adjusted crash risk. No “risk map,” driver score, crash-cause claim, or “lives saved” estimate is reported.

## Next steps before an engineering study

1. Obtain authorized, de-identified crash data with coverage documentation, consistent severity definitions, reliable event dates, and auditable geocodes.
2. Join crashes to a versioned road-segment network, retain match confidence, and manually review a stratified sample of snapped records.
3. Acquire matching-period traffic exposure (preferably vehicle-kilometres travelled) and road inventory/context. Road length alone is not traffic exposure.
4. Reconcile duplicate incidents across police, health, and news sources without collapsing legitimate co-located events.
5. Pre-register an exposure-adjusted count estimand and baseline; use spatial-block and future-period holdouts, calibration, uncertainty, and subgroup/coverage checks.
6. Annotate a permitted narrative sample with a codebook and dual-review subset before claiming NLP accuracy; keep an event-level holdout.
7. Verify image source/location metadata before any CV-to-road join. If no join can be justified, retain the visual benchmark as a separate road-condition model.
8. Present candidate locations only to qualified engineers with the supporting counts, exposure, match confidence, uncertainty, and field-inspection result. Any intervention evaluation needs a prospective design.

## Reproduction

From the project folder, after source files are present locally:

```powershell
uv sync
uv run python scripts/run_project.py
uv run streamlit run dashboard/app.py
```

For the independent India RDD2022 benchmark, install the optional CV libraries and acquire the selected India/train files:

```powershell
uv sync --extra vision
uv run python -m road_safety.vision download-india
uv run python scripts/run_project.py --with-vision
```

The default vision run is a compact one-epoch fine-tune over capped training/validation subsets. `uv run python -m road_safety.vision train --full-run` uses the full grouped split for ten epochs and will take substantially longer. Review source terms and each report before sharing artifacts.

## Source references

- Ashutosh and Chand, [Media Reported Road Traffic Crash Data, Mendeley Data v7](https://data.mendeley.com/datasets/bc5sv6wnd9/7), DOI `10.17632/bc5sv6wnd9.7`.
- Ministry of Road Transport and Highways, [Road Accidents in India 2023](https://www.morth.gov.in/sites/default/files/Road-Accident-in-India-2023-Publications.pdf).
- RDD2022, [Figshare dataset record](https://doi.org/10.6084/m9.figshare.21431547.v1); check challenge-side license conditions for the exact archive.
