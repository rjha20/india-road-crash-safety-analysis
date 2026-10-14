# Hosted dashboard deployment and operations

## Current deployment shape

The Streamlit Community Cloud app runs `dashboard/app.py` from this repository's `main` branch. It is a public, retrospective portfolio dashboard over versioned aggregate artifacts committed under `data/processed/`. The app does not connect to a live crash feed, authenticate to government systems, ingest source text, load road images, or serve model predictions.

Runtime packages are pinned in the root `requirements.txt` and Python is pinned to 3.12 in `.python-version`. The broader analysis environment is managed separately with `uv.lock` and `pyproject.toml`.

## Clean-clone data boundary

The hosted dashboard must work without source downloads. Before publishing an update, confirm that every file read by `dashboard/app.py` is a tracked, aggregate artifact. In particular:

- The reported-event map uses `data/processed/reported_event_grid_public.csv`, grouped by year, state, and 0.25-degree cell.
- Each published year/state/cell group contains at least five sampled events. Lower-support groups are removed before the file is written.
- The public grid excludes exact coordinates, crash dates, location descriptions, article text, and source row identifiers.
- `data/processed/crash_events_clean.csv`, raw workbooks, article text, source images, and fitted model weights stay local and gitignored.
- Published metrics/reports are aggregates, include source references, and retain the caveats about the selected media sample and non-operational model results.

The 0.25-degree map symbols are approximate cell centers, not road-segment locations. Filtering cannot reveal groups omitted from the published file.

## Create or restore the Streamlit app

1. Sign in to Streamlit Community Cloud with the authorized `rjha20` GitHub account.
2. Create an app from repository `rjha20/india-road-crash-safety-analysis`, branch `main`, and entrypoint `dashboard/app.py`.
3. Select Python 3.12. No app secrets are required.
4. Wait for the initial deployment to finish. Confirm the app loads the overview and all six tabs, then check the Cloud logs if startup fails.
5. Record the assigned app URL in this file and the README once verified.

Subsequent commits to `main` trigger a Streamlit rebuild. The dashboard is intended to require no credentials because it reads only committed aggregate files.

## Rebuild local analyses

The full reproducible workflow needs source files that are intentionally excluded from Git. Read `data/raw/README.md` and `docs/data_policy.md`, acquire each input under its own terms, verify hashes in `data/raw/source_manifest.json`, then run:

```powershell
uv sync --locked
uv run python scripts/run_project.py
```

The command regenerates the event-level extract locally and also writes the suppressed public grid. Review every changed artifact before committing; never add `crash_events_clean.csv`, raw news text, road images, or fitted weights. Optional CV training requires the separate `vision` dependency extra and the locally obtained image subset.

## Release verification

For each dashboard update:

1. Check `git status` and `git diff --check`.
2. Run Python syntax compilation for changed modules and the relevant data preparation command.
3. Confirm the public grid schema contains only `year`, `state`, `cell_latitude`, `cell_longitude`, `reported_events`, and `recorded_fatalities`; verify every grouped count is at least five.
4. Inspect the Streamlit page after the cloud rebuild. Confirm the aggregate map and the official statistics, article-theme, conditional-ML, and road-image tabs render.
5. Confirm the UI still says historical/exploratory and does not call the media sample a population-risk estimate.

## Operational limits and incident response

This is a public portfolio deployment, not a live safety service. It has no source freshness monitor, scheduled ingestion, API, auth boundary, alerting, production data store, SLA, current incident intake, or validated road-segment risk model. The dashboard's values change only when reviewed aggregate artifacts are regenerated and published. A failed or stale upstream source therefore cannot create a false “live” signal; the page must continue to describe the actual source vintages.

If the app fails to start, use Streamlit Cloud's deployment logs and restore the last working commit if needed. If a release accidentally contains a restricted source file or row-level artifact, stop further deployments, remove it from the current branch, and follow the repository host's sensitive-data removal process for its history and caches before redeploying.
