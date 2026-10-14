# India Road-Crash Hotspot and Hazard Analysis

## End-to-end project blueprint

**Prepared:** 7 October 2026  
**Purpose:** A senior-level implementation plan combining statistics, ML, deep learning, NLP, computer vision, and geospatial analysis.

## 1. Problem and decision

Road-safety engineers have limited time to inspect road sections. This project builds an evidence-based shortlist of locations for **human engineering review** by combining public crash reports, official aggregates, road-network context, weather, narrative text, and road-damage imagery.

The supported decision is: **Which mapped road sections should a qualified engineer investigate next, and what evidence supports each candidate?**

A model ranking is exploratory. It cannot substitute for field inspection, an official black-spot determination, or engineering judgment. It must not directly trigger enforcement, penalties, road closure, or construction.

### Two implementation tracks

India has public annual crash statistics, selected state black-spot reports, and media-derived crash records. These do not provide a complete open census of geocoded crashes joined to traffic exposure. MoRTH describes e-DAR as a national crash repository, but the public demo is not an openly downloadable research dataset. A 2024 road-safety report discusses police underreporting and the need for researcher access to e-DAR. ([MoRTH/PIB](https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=2110971&lang=2&reg=48), [India Status Report on Road Safety 2024](https://tripc.iitd.ac.in/assets/publication/India_Status_Report_on_Road_Safety-20242.pdf))

1. **Public-data demonstrator:** Analyze locations of collected/media-reported fatal crashes, compare patterns with official aggregates/black-spot reports, extract fields from narratives, and detect road damage in a separate labeled image collection. Conclusions must say “reported-event pattern” or “candidate for review,” not national crash probability or causal hazard.
2. **Agency-data extension:** Only after authorized access to representative record-level crashes and traffic exposure, estimate segment-period crash counts/rates and validate future-period predictions. If data access is unavailable, keep the public demonstrator scope and state that population crash risk was not modeled.

### Stakeholders and non-goals

Stakeholders: road-safety engineers/highway authorities, transport planners, public-health analysts, police and emergency-service data stewards, researchers, and public readers of aggregated results.

Non-goals: driver or neighborhood risk scores; real-time dispatch/navigation; individual blame; automated intervention decisions; causal claims based on observational correlations; national representative estimates from news coverage.

## 2. Outcomes, estimands, and KPIs

| Track | Estimand | Supported conclusion |
|---|---|---|
| Public-data demonstrator | Spatial concentration of collected/media-reported fatal crashes; NLP extraction accuracy; CV performance on its labeled source | Where the sampled reports cluster and what evidence merits expert review |
| Agency extension | Expected crash/severe-crash count per segment-period at a specified traffic exposure | Estimates for the population, time, geography, and outcome represented in authorized data |

Counts are not risk. A fatal-only media sample has no denominator for all crashes and reflects editorial choices. OSM road length is context, **not** VKT or traffic volume. Without exposure, report counts/density must never be labeled comparative risk.

### KPI plan

- Record source URL/DOI, version/hash, retrieval date, license, geography, time span, and transformations for every dataset.
- Before crash-level modeling, require usable date and state on at least 90% of in-scope rows; measure geocoding coverage and error rather than assuming it.
- Audit a stratified sample of location-to-road matches and report match distance/confidence.
- Compare hotspot methods against raw-count baselines; show uncertainty intervals, rank stability under bootstrap resampling, and support counts.
- For authorized crash-count models, report count deviance/log score, calibration, precision-recall AUC for rare severe events, recall/precision at a fixed inspection capacity (such as top 50 segments), and regional calibration. Do not rely on AUROC alone.
- For public media data, title any severity model **conditional media-sample severity analysis**. It is not a crash occurrence or road-risk model.
- CV: per-class precision/recall/F1 and mAP@0.5:0.95 on a location/source-grouped holdout; disclose India-subset size and class balance.
- NLP: field/entity precision/recall/F1 on a manually reviewed, crash-grouped holdout; report geocoding success separately from extraction accuracy.
- Do not claim lives saved or fewer crashes unless a prospective controlled intervention evaluation is later performed.

## 3. Verified public data inventory and caveats

Check and save terms at retrieval time. Publicly visible does not always mean reusable. OGD resource-level license metadata governs each resource; GODL-India requires attribution and excludes personal information or data the provider cannot license. ([OGD terms](https://www.data.gov.in/terms-of-use), [GODL-India](https://ap.data.gov.in/godl))

| Source | Access, use, license, and limitations |
|---|---|
| [MoRTH, Road Accidents in India 2023](https://www.morth.gov.in/sites/default/files/Road-Accident-in-India-2023-Publications.pdf) | Public annual report; state/UT and selected National Highway aggregate tables. Attribute MoRTH and check reuse terms. It is not a geocoded event feed; publication lag and category changes limit temporal comparison. Use for descriptive context/reconciliation. |
| [OGD, Road Accidents in India 2019](https://www.data.gov.in/catalog/road-accidents-india-2019) | MoRTH catalog lists 60 resources, largely state/city/road-category tables and selected 2016–2019 series; CSV/ZIP/API links and license metadata are available. Historical and mostly aggregate, with varying resource definitions; no exposure-complete road-segment panel. |
| [Maharashtra Highway Traffic Police, Black Spot 2021–2023](https://highwaypolice.maharashtra.gov.in/publication/maharashtra-black-spot-2021-2023/) | Official page links a 2021–23 black-spot PDF. Public page; verify the document’s reuse terms and attribute it. It is state-specific and selected for high-severity criteria, not a complete crash panel or a “safe road” control sample. |
| [e-DAR demo](https://edardemo.parivahan.gov.in/) and [MoRTH description](https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=2110971&lang=2&reg=48) | Repository is described as central accident reporting/analysis infrastructure. No open bulk research export was verified. Treat record-level access as permission-gated; do not scrape authenticated systems. Coverage, completeness, and refresh cadence require confirmation. |
| [Mendeley Media Reported Road Traffic Crash Data v7](https://data.mendeley.com/datasets/bc5sv6wnd9/7), [data paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12098169/) | CC BY 4.0 per Mendeley; cite authors and DOI 10.17632/bc5sv6wnd9.7. The paper describes 2,898 fatal crashes reported by Times of India in 2022–23, with dates, locations/coordinates, road type, vehicle pair, fatalities/injuries, and raw narrative text. English-language, fatal-only, media-selected; duplicates, geocoding ambiguity, and editorial bias are material. Do not republish full news article bodies; review rights for underlying newspaper text. Not a representative police dataset. |
| [OpenStreetMap copyright/license](https://www.openstreetmap.org/copyright/en) | India-wide road centerlines and mapped context under ODbL. Attribute “© OpenStreetMap contributors,” explain ODbL and review share-alike implications for adapted databases. Coverage/tags vary; road length is not traffic exposure. Use extracts and a suitable tile provider, not bulk calls to public tile/Nominatim services. |
| [RDD2022 road-damage images](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547), [paper](https://arxiv.org/abs/2209.08538) | CC BY 4.0; cite DOI 10.6084/m9.figshare.21431547.v1. 47,420 images from six countries including India, with over 55,000 annotations for longitudinal/transverse/alligator cracks and potholes. 2022 collection; not a live road inventory, and images are not linked to crashes by time/location. Check whether India-subset metadata supports geographic holdout. |
| [Copernicus ERA5 hourly weather](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=download), [product guide](https://confluence.ecmwf.int/pages/viewpage.action?pageId=576394630) | Hourly reanalysis from 1940 to present, time-series product on 0.25° grid; current catalog lists CC-BY. CDS account/API and acceptance of dataset terms required. Reanalysis is tens-of-kilometres scale, not roadside observation; preserve query, version, timestamp, and timezone. Rainfall does not establish fog/visibility. |
| State CCTNS/CID FIR portals; [India Status Report methodology](https://tripc.iitd.ac.in/assets/publication/India_Status_Report_on_Road_Safety-20242.pdf) | Potential narrative/structured data where states publish downloadable FIRs; report describes study extraction from six states. Access and terms vary. Confirm privacy/reuse with data stewards; remove personal identifiers. Public availability alone does not settle reuse. Use Mendeley for public NLP if access is unclear. |

### Source selection gate

Choose geography only after checking real overlap. Maharashtra is a candidate because of its official recent black-spot report, not a fixed assumption. Preserve a source manifest and hashes. If record-level data or VKT is unavailable, deliver a descriptive demonstrator; never synthesize crash records to imply national coverage.

## 4. Data model and geospatial processing

Core tables:

- **source_manifest:** provider, URL/DOI, version, retrieval date, terms, checksum, geography, period, transformation version.
- **crash_event:** project event ID, source IDs, event date/time and timezone confidence, original location phrase, coordinates/provenance, severity, vehicles/collision, source, deduplication group, quality flags.
- **road_segment:** stable segment ID, geometry, OSM version/tags, length, snap confidence.
- **segment_period:** segment/time window, count/fatalities/injuries, exposure plus source if available, covariates, source coverage, uncertainty.
- **weather_cell_hour:** ERA5 cell/time, units, variables, retrieval/version.
- **road_image and detection:** image ID, dataset/split/country/license, labels/detections, model version/confidence; coordinates only when actually supplied.
- **narrative_annotation:** event group, redacted text reference, human gold labels, reviewer flags, extraction/confidence.

Keep raw sources immutable and license-aware; transformed tables in Parquet; model-ready features separately. Maintain row-level provenance. Do not place names, phone numbers, registration numbers, FIR numbers, or household locations in analytic outputs.

Processing sequence: normalize WGS84 while preserving originals; validate/swap coordinates only under documented rules; manually audit a stratified geocode sample; deduplicate with reversible uncertain matches; snap to plausible roads under a network-aware threshold; record distance/confidence/unmatched rows; define segment length and period before modeling; sensitivity-test alternate segment sizes. Do not join images to crashes without verified time and location.

**Key cautions**

- **Sparse events:** small counts produce unstable ranks. Use partial pooling/empirical-Bayes smoothing, intervals, minimum-support rules, and bootstrap rank stability.
- **Exposure:** high-volume roads naturally accumulate more events. Prefer VKT (traffic volume × segment length × period) offset for rates. Road kilometres/population are proxies, not equivalent.
- **Ecological fallacy:** area averages do not reveal individual behavior or individual risk.
- **Police reporting bias:** reporting varies by severity, geography, access, and year; undercount is possible. Map source coverage/missingness.
- **Causality:** observational patterns are hypotheses for review, not proof of road design or weather causation.

## 5. Implementation phases: zero to handoff

### Phase 0 — Scope and access gate (1 week)

Define user, inspection decision, non-goals, geography candidates, source/license register, privacy rules, and source-to-claim matrix. Confirm geocoded events plus a matching reference source exist. If not, switch to a documented state/aggregate case study. Keep agency risk modeling blocked until representative records and exposure are authorized.

### Phase 1 — Reproducible repository and ingestion (1–2 weeks)

Set up Python 3.11/3.12, uv lockfile, typed config, structured logging, .env.example, source manifest, and data/license documentation. Implement downloads/imports with checksums, retry/backoff, schema checks, and API limits. Exclude large/restricted data and secrets from Git. Standardize dates, units, names, missing values, and source identifiers. Preserve raw records.

**Exit:** clean setup can reproduce manifests and row counts using documented files/manual-download steps.

### Phase 2 — Data QA and EDA (1–2 weeks)

Measure duplicates, coordinate errors, missing dates, category drift, coverage by geography/source/year, and point-to-road match quality. Audit geocoding manually. Plot counts by source, state, month, severity, and road type. Reconcile official aggregates only for matching definitions/years; document discrepancies rather than forcing them to agree.

**Exit:** quality report states numerator/denominator, missingness, matching accuracy, reporting limits, and source coverage.

### Phase 3 — Statistics and hotspot baselines (1–2 weeks)

Public track: event counts, network-constrained density/KDE, global Moran’s I and local Moran/Getis–Ord with permutation tests and multiple-testing correction; empirical-Bayes or hierarchical Poisson/negative-binomial smoothing for sparse counts; interval and sensitivity analysis across segment length, snap threshold, and deduplication.

Agency track: segment-period Poisson/NB with log(VKT) offset; test overdispersion, zero frequency, residual spatial autocorrelation, temporal drift, and offset quality. Benchmark against simple rates/counts.

**Exit:** maps and memo clearly separate concentration from exposure-adjusted rate.

### Phase 4 — Structured ML (1–2 weeks)

Public data may support **fatality-count/severity-band prediction conditional on a media-reported fatal crash**, using only available non-leaking covariates (road type, vehicle pair, time, geography, validated weather). Compare negative-binomial/regularized GLM with tree boosting. Exclude target-derived narrative fields, article-publication leakage, and post-event features. Calibrate and report feature attribution as descriptive, not causal.

This is a method demonstration in a selected media sample—not crash occurrence or road risk. Agency extension may model representative segment-period counts/severe counts with VKT offset, using time-available covariates only. Use review-capacity thresholds, not universal cutoffs.

**Exit:** model card defines population, target, split, calibration, coverage, limitations, and score-withholding rules.

### Phase 5 — Deep learning/CV (1–2 weeks)

Use the India subset of RDD2022 for pothole/crack detection. Inspect labels, imbalance, duplicates, and source grouping. Establish pretrained torchvision detector baseline; confirm weight/data licenses before fine-tuning. Split by location/sequence/source when possible, not random image rows. Report per-class metrics and errors; assess lighting, blur, material, and class variation if available. Do not call detections crash causes.

**Exit:** reproducible India-subset CV report; disclose if location metadata is insufficient for a true geographic holdout.

### Phase 6 — NLP narratives (1–2 weeks)

Use Mendeley narrative fields only within dataset and underlying-text terms; do not publish article bodies. Annotate a stratified 300–500 examples if practical; write guidelines and measure inter-annotator agreement on a subset. Extract time/place/road/vehicles/collision/casualties/weather/negation using rules first, then spaCy or a compact transformer, with explicit unknown/not-stated and abstention. Human-review ambiguous place resolution. Split by crash and near-duplicate group before training/testing.

**Exit:** field-level F1, human gold definition, geocoding confidence, errors, and correction path documented.

### Phase 7 — Fusion and leakage-resistant validation (1–2 weeks)

Join official aggregates only by matching geography/year; weather by coordinate/time; road context by network match; narratives to their originating event. Join CV only with reliable image time/location. Keep missing evidence distinct from “no hazard/no crash.” Display separate evidence layers; if a combined score is used, show weights and missingness behavior.

Deduplicate before splitting. Hold out a future period and whole districts/corridors with spatial buffers; use temporal embargo for overlapping windows. Fit preprocessing and calibration on train folds only. Group NLP by event/source and CV by image sequence/location. Never random-split neighboring cells or repeat articles across folds. A two-year news sample cannot prove future-year generalization; say so. Check errors/rank stability across urban/rural, road class, state, source, and coverage strata; suppress unstable estimates.

Use bootstrap/posterior intervals, calibration, prediction intervals, and rank stability. Explanations are hypotheses for engineers. Show event count/exposure, source coverage, match confidence, uncertainty, context evidence, and model version.

### Phase 8 — Dashboard/API workflow (1–2 weeks)

Build a Streamlit map/dashboard with year/source/road-class filters, count vs rate toggle (disable rate without exposure), uncertainty, data coverage, and source drill-through. Visually separate crashes, official black spots, roads, weather, CV detections. Include OSM attribution/ODbL. Avoid raw personal data/small identifiable counts. Every result view states “exploratory; engineer review required.”

FastAPI is optional only when there is a consumer; expose aggregate, versioned endpoints and no raw FIR/article endpoint. A reproducible dashboard is sufficient MVP.

**Exit:** analyst can filter results, inspect evidence/uncertainty, and export a sourced review shortlist.

### Phase 9 — Deployment, monitoring, handoff (1 week)

Containerize local/demo deployment and pin dependencies. Publish only redistributable data; otherwise give manifests/download instructions and tiny synthetic fixtures. Monitor source freshness/schema drift, missing coordinates, geographic coverage, category changes, calibration when new labels arrive, and reviewed CV/NLP error. Human workflow: candidate → analyst triage → engineering field verification → decision outside model. Retrain only when labels and validation gates justify it.

**Exit:** clean-clone reproduction succeeds; safety wording reviewed; missing source failures are visible.

## 6. Architecture and repository

**Stack:** Python, uv, pandas/Polars, PyArrow/Parquet, DuckDB; PostgreSQL/PostGIS for shared deployment only; GeoPandas/Shapely/pyproj/OSMnx; statsmodels, PyMC, scikit-learn, LightGBM/XGBoost; PyTorch/torchvision/Albumentations; spaCy/Transformers; Streamlit/Plotly/pydeck; FastAPI optional; Docker; pytest/Ruff; DVC or checksums for large files.

~~~text
road-safety-india/
  README.md  pyproject.toml  uv.lock  .env.example
  configs/  data/{raw,interim,processed,fixtures}/
  docs/{problem,source-license-register,data-dictionary,quality,validation,model-cards,privacy,runbook}/
  src/road_safety/{ingest,quality,geospatial,features,stats,models,evaluation,dashboard}/
  notebooks/  sql/  tests/{unit,data-contracts,geospatial,leakage,integration}/
  reports/{figures,baseline,statistical-analysis,model-evaluation,final-case-study}/
  deployment/{Dockerfile,compose.yaml}
~~~

Pipeline: source manifest → immutable inputs → schema/quality checks → normalized events/roads → geo linkage and exposure/context joins → statistical baselines → ML/NLP/CV → spatial-temporal evaluation → aggregate evidence → reviewed dashboard → monitoring/handoff.

## 7. Reproducibility and test strategy

Use pinned dependencies, fixed seeds, source hashes, saved split IDs/config, model version, and report build instructions. Tests should use tiny synthetic or redistributable fixtures—not full downloads.

- Unit: coordinate/timezone/unit/severity conversion, deduplication, snapping, offsets, uncertainty/rank logic.
- Contracts: required columns, domains, null thresholds, keys, source version.
- Geospatial: known point-to-road/district, CRS, geometry, distance thresholds.
- Leakage: no event/duplicate/image sequence/spatial block crosses splits; targets/post-event fields excluded.
- Statistical: deterministic toy counts verify aggregation, offset, interval and sparse behavior.
- NLP/CV: fixture parsing/inference; frozen human labels for metric evaluation.
- Integration: fixture through normalized data to model metrics/map aggregates.
- Report final metrics once on frozen test set; if sample is too small, use grouped repeated CV with uncertainty and disclose instability.

## 8. Acceptance criteria

1. Clear decision, user, geography, non-goals, and engineer-review requirement.
2. Source/license register includes exact versions, retrieval dates, license, age, coverage, and use decisions.
3. QA reports duplicate rate, geocoding accuracy, road-match coverage, missingness, and representativeness.
4. Counts, rates, and exposure are distinct; statistical baseline, intervals, and sensitivity checks exist.
5. ML target is valid, leakage-controlled, spatial-temporal holdout tested, calibrated, baseline-compared, and documented in a model card.
6. CV includes class errors and India subset metrics with grouped split where metadata permits.
7. NLP includes human labels, group split, field-level scores, geo confidence, and abstention/correction.
8. Fusion only uses defensible joins; no unsupported defect-to-crash causal claim.
9. Dashboard has provenance, definitions, uncertainty, coverage, license attribution, no personal identifiers, and safety wording.
10. Clean clone reproduces fixture pipeline; protected data/credentials are not embedded.
11. Rankings are exploratory and require qualified engineering review.
12. If representative crash data/VKT is unavailable, state that population-level crash-risk modeling was not done.

## 9. Risks and controls

| Risk | Control |
|---|---|
| No complete geocoded crash record set | Access gate; public demonstrator limited to collected reports; never invent labels |
| Missing traffic exposure | VKT offset for agency model; otherwise counts/density only |
| Police underreporting varies | Map coverage/missingness; compare sources carefully; no simplistic state ranking |
| Media bias | Limit claims to media-reported events; no prevalence estimates |
| Sparse counts/rank instability | Shrinkage, intervals, bootstrap ranks, minimum support |
| OSM incompleteness/geocoding errors | Match confidence, manual audit, unmatched records retained |
| CV domain shift/nonlinked images | India subset evaluation; keep separate layer; no causal or location join without metadata |
| NLP ambiguity/privacy | Gold review, abstention, redaction, permission/terms review |
| Leakage | Deduplicate first; spatial/time/group holdouts and split audits |
| Model misuse | Exploratory labels, uncertainty, human field review, no automated action |

## 10. Timeline and effort

A solo polished portfolio version is approximately **240–320 person-hours**, or **10–16 weeks at 20–24 hours per week**. Main risks are geocoding, data cleanup, deduplication, and source access.

| Workstream | Hours |
|---|---:|
| Scope and source/license audit | 16–24 |
| Environment, manifests, ingestion | 24–32 |
| QA, geospatial linkage, EDA | 32–44 |
| Statistical baselines | 24–32 |
| Structured ML and validation | 24–32 |
| CV and evaluation | 28–40 |
| NLP annotation/extraction/evaluation | 28–40 |
| Fusion and leakage audit | 20–28 |
| Dashboard, reporting, runbook | 24–32 |
| Packaging, checks, final review | 20–28 |
| **Total** | **240–332** |

Estimate excludes e-DAR agreements, large FIR transcription, professional traffic counts, new camera surveys, and intervention evaluation. An agency-grade pilot needs roughly **6–9 calendar months after data agreements**, plus field validation and prospective evaluation before operational use.

If time is limited: source/QA → statistical baseline → NLP → CV → conditional severity analysis → dashboard. The work remains worthwhile if the exposure-data gate prevents risk modeling.

## 11. Handoff package

- Reproducible repository and run instructions.
- Data dictionary, source/license register, manifests, QA report.
- Statistical analysis with counts, exposure definitions, intervals, sensitivity checks.
- Separate ML, CV, and NLP evaluations/model cards.
- Interactive candidate-inspection dashboard, privacy/use note, and engineering review policy.
- Final case study and one-page e-DAR/agency data request specification: crash time, geocodable location, severity, road identifiers/class, report status, de-identification, sampling/coverage, and VKT/traffic volume.

**Success is a transparent, defensible inspection shortlist and reproducible analysis—not an algorithm claiming to know why crashes occurred.**

