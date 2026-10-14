# NLP and structured model evaluation

## Narrative NLP: 2022 raw article corpus

- **Coverage:** 1,565 unique nonempty articles from the local 2022 raw CSV, after removing 3 exact duplicate article bodies for aggregate counts.
- **Observed columns:** title `Title`, date `Date`, content `content`.
- **Article-date range:** 2022-01-01 to 2022-12-31.
- **Coverage limitation:** only the 2022 raw file was available. The attempted 2023 raw download was a 395-byte internal-error JSON response and was rejected as data.

We ran transparent dictionary-pattern extraction across title and content. The counts below are **articles containing a keyword theme**, not verified crash factors. Themes can overlap; negation, attribution, and temporal sequence are not resolved. Article-level matches and text are not exported. No hand-labeled gold sample was available, so precision, recall, F1, and NLP model accuracy are intentionally not reported.

| Weak mention theme | Articles | Share of unique 2022 articles |
|---|---:|---:|
| `heavy_vehicle_mentioned` | 829 | 52.97% |
| `two_wheeler_mentioned` | 604 | 38.59% |
| `speed_or_unsafe_manoeuvre` | 600 | 38.34% |
| `hit_and_run_mentioned` | 126 | 8.05% |
| `pedestrian_or_cyclist_mentioned` | 97 | 6.20% |
| `helmet_mentioned` | 79 | 5.05% |
| `weather_or_visibility` | 57 | 3.64% |
| `alcohol_mentioned` | 46 | 2.94% |
| `wrong_side_or_lane` | 45 | 2.88% |
| `road_surface_or_design` | 42 | 2.68% |
| `seat_belt_mentioned` | 1 | 0.06% |

These counts describe coverage of a newspaper corpus of fatal-crash stories. They cannot estimate prevalence among all crashes or prove that a mentioned factor caused a crash. For valid NLP evaluation, annotate an event-grouped sample with a written codebook, dual-label a subset, report agreement, freeze an article/event-grouped holdout, and score entity/attribute extraction against human gold labels.

Raw titles and article bodies remain under `data/raw/mendeley/v7/raw_articles/` locally. Processed NLP outputs contain only theme-level counts and source metadata. Do not publish raw article text or named entities.

## Structured ML: conditional fatality-band demonstration

**Question actually modeled:** among this dataset's media-reported fatal crashes, can a small set of source-coded roadway/collision/vehicle attributes distinguish rows with one recorded death from those with two or more recorded deaths?

The dataset has 2,898 usable rows: 1,442 with one death and 1,456 with two or more. Positive class prevalence is 50.2%. The binary cutoff is a simple, predeclared demonstration label, not an official severity standard.

### Validation design

- Five-fold (or fewer if there are fewer state groups) stratified grouped cross-validation; complete states are held out together. The reported pooled metrics are from out-of-fold predictions.
- Fixed logistic regression with one-hot categorical encoding, rare-category minimum frequency 2, and balanced class weights. No hyperparameter search was run.
- Predictors: `road_type, crash_type, vehicle_1, vehicle_2`.
- `State` is used only to keep geographic groups apart; it is not a predictor. `Injured`, article publication date, age/gender, coordinates/location, serial number, and narrative text are excluded.
- A majority-class `DummyClassifier` is the baseline. The 0.5 cutoff is diagnostic only; it is not an action threshold.

| Metric | Grouped OOF model | Majority baseline |
|---|---:|---:|
| Accuracy | 0.684 | 0.455 |
| Balanced accuracy | 0.684 | 0.455 |
| Precision (0.5) | 0.697 | 0.441 |
| Recall (0.5) | 0.655 | 0.319 |
| F1 (0.5) | 0.675 | 0.370 |
| ROC-AUC | 0.746 | 0.455 |
| Average precision | 0.742 | 0.483 |
| Brier score (lower better) | 0.204 | 0.545 |

Pooled model confusion matrix with rows=true `[one death, 2+ deaths]` and columns=predicted `[one death, 2+ deaths]`: `[[1027, 415], [502, 954]]`.

### Fold stability

| Fold | Held-out rows | Held-out states | F1 | ROC-AUC | Average precision |
|---:|---:|---:|---:|---:|---:|
| 1 | 515 | 7 | 0.712 | 0.809 | 0.762 |
| 2 | 627 | 5 | 0.666 | 0.707 | 0.750 |
| 3 | 614 | 8 | 0.664 | 0.754 | 0.764 |
| 4 | 605 | 6 | 0.693 | 0.750 | 0.759 |
| 5 | 537 | 4 | 0.644 | 0.730 | 0.680 |

Cluster bootstrap intervals in `data/processed/conditional_fatality_band_metrics.json` resample state groups from pooled out-of-fold predictions; they measure sample instability, not population coverage or external validity.

## Model card and use limits

| Item | Description |
|---|---|
| Model | One-hot encoded, class-weighted logistic regression; fitted artifact saved at `data/processed/conditional_fatality_band_logistic.joblib`. |
| Intended use | Reproducible portfolio demonstration of structured modeling and grouped validation among the source's already-fatal reported events. |
| Prohibited interpretation | Do not use as a road hotspot, crash occurrence, individual, causal, emergency-response, enforcement, insurance, infrastructure funding, or engineering action score. |
| Training/evaluation data | Mendeley v7 Times of India media-reported fatal crashes; English newspaper source; observed period 2022-01-01–2022-12-31 for articles if available. Structured event range is in `reports/data_quality.md`. |
| Target | `Killed >= 2`, conditional on inclusion in this fatal-crash media sample. |
| Validation | Grouped by state; no future-year or external-source validation. |
| Known bias | Fatal-only selection, newsworthiness/editorial coverage, location/source representation, extracted-field error, sparse categories, and no all-crash or traffic-volume denominator. |
| Calibration | Brier score is reported, but probabilities are not calibrated for decisions; do not set operational cutoffs. |
| Monitoring needed before any future authorized use | Source/schema drift, state and road-class coverage, target definitions, missingness, groupwise calibration and error, independent representative labels, and traffic exposure. |

**Conclusion:** this experiment can demonstrate preprocessing, a leakage-aware target, group holdout, baseline comparison, and honest reporting. It cannot establish which Indian road segments are dangerous. A defensible hotspot/risk model needs representative police/health crash records linked to a road network plus traffic exposure (for example vehicle-kilometres travelled), followed by spatial and future-period validation.
