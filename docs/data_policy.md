# Data-use and interpretation policy

1. **Scope:** This repository is an educational/portfolio research prototype for historical analysis and model evaluation.
2. **No operational use:** It must not issue crash alerts, direct enforcement, decide road works, determine compensation, or assign blame.
3. **Source populations:** Official MoRTH aggregates, TOI-reported fatal event records, 2022 article text, and RDD2022 images have different populations. They remain distinct unless a defensible record-level join is demonstrated.
4. **Text rights:** Store raw article text locally and outside version control. Extracted concepts and aggregate counts may be included only after review; never publish article bodies or long quotations.
5. **Personal data and public map:** Do not expose victim names or row-level age/gender/location profiles. Keep `crash_events_clean.csv` local and ignored. The hosted map may use only `reported_event_grid_public.csv`, grouped by year, state, and 0.25-degree cell; suppress each group below five events before release. Include aggregate counts and approximate cell centers only.
6. **Model language:** Use “reported-event sample,” “conditional on a sampled fatal crash,” “visual benchmark detection,” and “candidate for engineer review.” Do not use “probability this road is dangerous,” “causes,” or “lives saved.”
7. **Missingness:** Missing coordinates, article text, or image coverage are unknown/coverage gaps, not evidence of no crash or no hazard.
8. **Attribution:** Include Mendeley dataset DOI, MoRTH report citation, RDD2022 DOI and license terms in the final project. Retain OSM attribution only if OSM data is actually used.
9. **Release gate:** Before sharing files, verify all data/model licenses; remove raw and restricted files; check that aggregate tables/figures do not reveal person-level records; state validation and coverage limits. For the hosted dashboard, confirm every runtime input is committed and aggregate-only.
