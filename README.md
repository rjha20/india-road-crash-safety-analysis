# India Road-Crash Hotspot and Hazard Analysis

A reproducible data-science portfolio project that studies **where media-reported fatal crashes cluster, what patterns appear in their structured records and narratives, and what road-surface damage a separate India-focused vision benchmark can detect**.

> **Decision boundary:** This is exploratory research. It does not estimate population crash risk, identify causal crash factors, issue alerts, or replace road-safety engineers or field inspections. Counts from news reports are not exposure-adjusted risk. The road-image benchmark is not linked to the crash events.

## Project components

1. **Statistics / geospatial analysis:** summarize official MoRTH state and selected National Highway aggregates; analyze coarse cell concentrations and haversine DBSCAN groups in the non-representative fatal-crash sample; quantify count uncertainty and rank stability.
2. **Machine learning:** evaluate whether source-coded attributes distinguish one from two-or-more recorded deaths, conditional on an event already being a media-reported fatal crash. This is not a model of whether a road will have a crash.
3. **NLP:** extract or summarize safety themes from the locally stored 2022 article text using transparent baselines. Text is excluded from published outputs; 2023 source text could not be fetched when this build ran.
4. **Deep learning / computer vision:** train/evaluate a road-damage detector on the India subset of RDD2022. Images are an independent benchmark and must not be joined to crash records without location/time evidence.
5. **Review dashboard and case study:** present source coverage, limitations, aggregate counts, model performance, uncertainty, and the independent CV benchmark metrics with clear provenance.

The detailed rationale, source inventory, and complete phase plan are in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Hosted dashboard

Open the [live Streamlit dashboard](https://india-road-crash-safety-analysis-lffexwhoskco8qnavbxl9r.streamlit.app/). It runs `dashboard/app.py` from the `main` branch and reads only aggregate artifacts committed under `data/processed/`; the local row-level event extract, raw article text, images, and model weights are not deployed. The event map uses the k-suppressed `reported_event_grid_public.csv` artifact. See [deployment and operations](docs/deployment.md) for setup and verification.

This is a hosted retrospective research interface, not a live crash feed or operational road-safety system. Its sources are historical and cannot support exposure-adjusted road-segment risk estimates or safety decisions.

## Data and rights

| Source | Use | Important boundary |
|---|---|---|
| [Mendeley: Media Reported Road Traffic Crash Data, v7](https://data.mendeley.com/datasets/bc5sv6wnd9/7), DOI `10.17632/bc5sv6wnd9.7` | Structured crash attributes for 2,898 fatal events reportedly covered by Times of India in 2022–2023; data dictionary | Dataset landing page labels the compilation CC BY 4.0. Underlying news copy may have separate copyright. Raw article files stay local and ignored by Git; reports contain only aggregate results and short non-infringing summaries. Media selection and fatal-only scope make this unsuitable for prevalence or population risk estimates. |
| [MoRTH, Road Accidents in India 2023](https://www.morth.gov.in/sites/default/files/Road-Accident-in-India-2023-Publications.pdf) | Official state-wise aggregate context, including selected NH breakdowns | Aggregate definitions and denominators must be preserved. These tables do not provide an exposure-complete segment panel. Verify reuse terms before redistribution. |
| [RDD2022, Figshare](https://doi.org/10.6084/m9.figshare.21431547.v1) | India-only road image benchmark for cracks/potholes | Figshare lists CC BY 4.0; challenge materials may state additional/share-alike conditions. Preserve attribution and review all relevant terms. India images are mainly from Delhi, Gurugram, and Haryana; they are not geotagged to the crash sample. |
| [MoRTH/e-DAR information](https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=2110971&lang=2&reg=48) | Documents official crash repository context | No public bulk e-DAR record export is included. The project does not scrape authenticated systems. |

All source snapshots are local research inputs. `data/raw/` is excluded from Git. See [data policy](docs/data_policy.md). A clean clone can reproduce the published project only after obtaining the source data under its own terms; no large or copyrighted source files are redistributed here.

## Environment

- Windows PowerShell or another supported shell
- Python 3.11 or 3.12
- `uv` for a locked environment
- Optional PyTorch/Torchvision for the vision component; GPU is helpful but not required for the compact benchmark run
- Streamlit provides the interactive hosted dashboard; `uv` remains the reproducible environment for rebuilding analyses locally

## Quick start

From this folder:

```powershell
uv sync
uv run python scripts/run_project.py --help
```

The end-to-end public-data workflow runs data quality, official statistics, spatial concentration, the 2022 NLP summary, and grouped structured modeling in dependency order:

```powershell
uv run python scripts/run_project.py
```

Add the optional CV packages before training the separate vision component:

```powershell
uv sync --extra vision
```

For optional transformer NLP experiments:

```powershell
uv sync --extra nlp
```

Do not train a transformer by default: the corpus is small and the raw article text is rights-sensitive. Start with the transparent token/phrase and TF-IDF baselines.

## Reproduce from raw inputs

1. Read [data/raw/README.md](data/raw/README.md) and verify each downloaded file against `data/raw/source_manifest.json`.
2. Place the Mendeley v7 Data Files workbook and dictionary under `data/raw/mendeley/v7/`. For the NLP extension, place the permitted/local 2022 article CSV under `data/raw/mendeley/v7/raw_articles/2022_raw_data.csv`. The 2023 raw endpoint was failing at build time; do not substitute the error JSON for source data.
3. Obtain the India-only RDD2022 archive from the official challenge release and extract into `data/raw/rdd2022/India/`. The vision script should accept the VOC layout and print dataset discovery results before training.
4. Download the official MoRTH report or use the versioned derived state table under `data/processed/` if its manifest is present.
5. Run the full public-data workflow with `uv run python scripts/run_project.py`. To include CV, install `uv sync --extra vision`, obtain the India RDD2022 image subset with `uv run python -m road_safety.vision download-india`, then run `uv run python scripts/run_project.py --with-vision`. The orchestrator profiles annotations, creates a sequence-proxy grouped split, then runs a capped one-epoch training/evaluation; choose `uv run python -m road_safety.vision train --full-run` for the longer run.
6. The hosted dashboard can be opened from the link above using the committed aggregate outputs. For a local rebuild, generate the outputs first and then launch the dashboard.

Runnable commands and outputs:

| Component | Command | Primary output |
|---|---|---|
| Structured data quality and release-safe map grid | `uv run python -m road_safety.crash_data` | `reports/data_quality.md`; local row-level extract (ignored) and `data/processed/reported_event_grid_public.csv` |
| Official statistical baseline | `uv run python -m road_safety.statistics --reps 10000` | `reports/statistical_analysis.md`; state/UT aggregate table |
| Spatial concentration | `uv run python -m road_safety.spatial` | `reports/spatial_hotspot_analysis.md`; coarse cells with fewer than five events suppressed |
| 2022 narrative NLP | `uv run python -m road_safety.nlp` | `reports/nlp_and_model_evaluation.md`; aggregate theme counts only |
| Conditional structured ML | `uv run python -m road_safety.crash_models` | `reports/nlp_and_model_evaluation.md`; grouped out-of-fold metrics and local fitted artifact |
| Computer vision | `uv run python -m road_safety.vision profile`; `uv run python -m road_safety.vision split`; `uv run python -m road_safety.vision train` (default capped; `--full-run` for full split) | `reports/vision_evaluation.md`; independent RDD2022 benchmark results, when the optional archive and dependencies are available |
| Dashboard | `uv run streamlit run dashboard/app.py` | Local copy of the hosted aggregate-only review interface |
| Full rebuild | `uv run python scripts/run_project.py` | Rebuildable non-CV outputs; add `--with-vision` only after installing the vision extra and obtaining the India subset |

## Data handling and modeling principles

- Keep the source row number only in the locally ignored structured extract for QA; omit victim age/gender values and all news text from derived rows.
- Do not display names, article bodies, or row-level crash/victim characteristics in the dashboard. The published event map uses coarse year/state cells with a five-event support threshold.
- Parse date and coordinate fields explicitly; keep invalid/unmatched records visible in QA counts rather than silently dropping them.
- Remove exact duplicate article bodies before corpus-level counts. For structured events, keep ambiguous duplicate signatures visible and report sensitivity rather than silently merging possible distinct crashes. The severity model holds out state groups; it has no future-year or external validation.
- Use grouped validation where source/geographic clusters matter; a future-year holdout should be added when the data and estimand support it. Never rely on a random row split as the sole evidence of generalization.
- Treat `Killed` as a conditional count within a fatal, media-reported sample. Do not call this “crash risk” or infer the chance of death for a road user.
- Separate official state totals from the Times of India event sample. They have different sources, coverage, and denominators.
- Separate RDD image detections from crash records. Do not construct a fused hotspot score from unrelated image and event observations.
- Report baselines, denominators, class distribution, uncertainty, coverage, and failure cases. An unstable or misleading model can be excluded while the remaining analyses still complete.

## Directory map

```text
PROJECT_PLAN.md               Senior-level end-to-end implementation plan
configs/project.yml           Sources, scope, reproducibility settings
src/road_safety/               Ingestion, QA, statistics, NLP, ML, CV modules
data/raw/                      Local inputs; ignored by Git
data/processed/                 Published aggregates and metrics; row-level extracts are locally ignored
reports/                        Data, method, evaluation, and case-study reports
dashboard/app.py                Hosted aggregate-only analyst-review dashboard
scripts/run_project.py          Full workflow orchestrator
```

## Interpretation and use

A location with many sampled articles is a **reported-event concentration**, not automatically an unsafe road. Official state aggregates may reflect exposure, reporting, definitions, and administrative coverage. RDD detections indicate a visual road-surface label in a benchmark image; they do not show a crash cause, maintenance priority, or crash location. The intended final product is a traceable research shortlist for qualified human review.
