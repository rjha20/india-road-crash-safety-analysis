"""Hosted, aggregate-only research dashboard for the India road-safety project."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"

st.set_page_config(page_title="India Road Safety | Analyst Review", page_icon="🛣️", layout="wide")


@st.cache_data
def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def read_csv(path: Path, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, usecols=list(columns) if columns else None)


def show_missing(message: str) -> None:
    st.info(message)


def metric_or_na(value: object, digits: int = 0) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Not available"
    if not math.isfinite(number):
        return "Not available"
    return f"{number:,.{digits}f}"


st.title("India Road-Crash Hotspot and Hazard Analysis")
st.caption("Hosted portfolio dashboard • historical source vintage: crash reports through 2023, MoRTH 2023, and RDD2022")
st.warning(
    "These sources describe different populations. MoRTH figures are official administrative totals. "
    "The event and article analyses use a media-selected sample of fatal crashes. Neither is an "
    "exposure-adjusted road-segment risk estimate. This is not a live incident feed or operational "
    "safety service. The road-image benchmark is a separate dataset."
)

tabs = st.tabs(
    ["Overview", "Official MoRTH", "Reported crash sample", "Article themes", "Conditional ML", "Road-image CV"]
)

# Overview: source populations remain separate even on the landing page.
with tabs[0]:
    quality = read_json(PROCESSED / "crash_quality_summary.json")
    metrics = read_json(PROCESSED / "conditional_fatality_band_metrics.json")
    nlp = read_json(PROCESSED / "nlp_summary_2022.json")
    morth_2023 = {"crashes": 480_583, "fatalities": 172_890}
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("MoRTH · all-road, 2023")
        st.metric("Reported crashes", f"{morth_2023['crashes']:,}")
        st.metric("Reported fatalities", f"{morth_2023['fatalities']:,}")
        st.caption("Official national administrative totals. See the MoRTH tab for state aggregates and NH tables.")
    with c2:
        st.subheader("Times of India sample · structured fatal events")
        st.metric("Sampled events", metric_or_na(quality.get("rows")))
        st.metric("Fatalities recorded in this sample", metric_or_na(quality.get("total_fatalities_in_sample")))
        st.caption("Mendeley v7 compilation; media-selected reports, not a census of Indian crashes.")
    st.divider()
    a, b, c = st.columns(3)
    a.metric("States/UTs in event file", metric_or_na(quality.get("states")))
    b.metric("Unique 2022 articles analyzed", metric_or_na(nlp.get("unique_articles_analyzed")))
    b.caption("Weak keyword-mention analysis; articles with unparsed dates are retained in corpus counts.")
    c.metric("Grouped-CV ROC-AUC", metric_or_na(metrics.get("out_of_fold_metrics", {}).get("roc_auc"), 3))
    c.caption("Predicts fatality band within the already-fatal media sample; not crash occurrence or road risk.")
    st.info(
        "A true road-segment hotspot model needs representative geocoded crash records, a road network, "
        "traffic exposure, and spatial plus future-period validation. These inputs are not present here."
    )

with tabs[1]:
    st.header("Official administrative baseline · MoRTH")
    st.write(
        "This panel uses the Ministry of Road Transport and Highways' *Road Accidents in India 2023*. "
        "It is separate from the Times of India event sample and contains state-level aggregates, not site coordinates."
    )
    totals = pd.DataFrame(
        [
            {"year": 2022, "measure": "All-road crashes", "count": 461_312},
            {"year": 2023, "measure": "All-road crashes", "count": 480_583},
            {"year": 2022, "measure": "All-road fatalities", "count": 168_491},
            {"year": 2023, "measure": "All-road fatalities", "count": 172_890},
            {"year": 2022, "measure": "National Highway crashes", "count": 151_997},
            {"year": 2023, "measure": "National Highway crashes", "count": 150_177},
            {"year": 2022, "measure": "National Highway fatalities", "count": 61_038},
            {"year": 2023, "measure": "National Highway fatalities", "count": 63_112},
        ]
    )
    fig = px.bar(totals, x="measure", y="count", color="year", barmode="group", title="Published national totals")
    fig.update_layout(yaxis_title="Reported count", xaxis_title="", legend_title="Year")
    st.plotly_chart(fig, width="stretch")

    state = read_csv(PROCESSED / "morth_nh_2023_state.csv")
    if state.empty:
        show_missing("The derived MoRTH state table is not present yet.")
    else:
        state = state.dropna(subset=["state_ut"])
        left, right = st.columns(2)
        with left:
            st.subheader("Top state/UT counts · all roads, 2023")
            top_crashes = state.dropna(subset=["accidents_2023"]).nlargest(10, "accidents_2023")
            chart = px.bar(top_crashes.sort_values("accidents_2023"), x="accidents_2023", y="state_ut", orientation="h", title="Reported crashes")
            chart.update_layout(xaxis_title="Reported crash count", yaxis_title="")
            st.plotly_chart(chart, width="stretch")
            top_fatalities = state.dropna(subset=["fatalities_2023"]).nlargest(10, "fatalities_2023")
            chart = px.bar(top_fatalities.sort_values("fatalities_2023"), x="fatalities_2023", y="state_ut", orientation="h", title="Reported fatalities")
            chart.update_layout(xaxis_title="Reported fatality count", yaxis_title="")
            st.plotly_chart(chart, width="stretch")
        with right:
            st.subheader("National Highway tables, 2023")
            st.caption("MoRTH publishes separate top-ten state lists. Blank means not listed, not zero.")
            nh_crashes = state.dropna(subset=["nh_accidents_2023"]).nlargest(10, "nh_accidents_2023")
            chart = px.bar(nh_crashes.sort_values("nh_accidents_2023"), x="nh_accidents_2023", y="state_ut", orientation="h", title="NH crashes · listed states")
            chart.update_layout(xaxis_title="Reported crash count", yaxis_title="")
            st.plotly_chart(chart, width="stretch")
            nh_fatalities = state.dropna(subset=["nh_fatalities_2023"]).nlargest(10, "nh_fatalities_2023")
            chart = px.bar(nh_fatalities.sort_values("nh_fatalities_2023"), x="nh_fatalities_2023", y="state_ut", orientation="h", title="NH fatalities · listed states")
            chart.update_layout(xaxis_title="Reported fatality count", yaxis_title="")
            st.plotly_chart(chart, width="stretch")
    st.caption(
        "Sources: MoRTH, Road Accidents in India 2023 (national totals); the extracted state table is "
        "data/processed/morth_nh_2023_state.csv. Administrative counts are not exposure-adjusted rates."
    )

with tabs[2]:
    st.header("Media-reported fatal-crash event sample")
    st.caption("Source population: 2,898 structured records compiled from Times of India reporting (Mendeley v7).")
    st.warning(
        "This map shows coarse concentrations in a media-reported fatal-crash sample. It does not rank "
        "dangerous roads or estimate crash risk; there is no traffic-exposure denominator."
    )
    cols = (
        "year", "state", "cell_latitude", "cell_longitude", "reported_events", "recorded_fatalities",
    )
    public_events = read_csv(PROCESSED / "reported_event_grid_public.csv", columns=cols)
    if public_events.empty:
        show_missing("The shareable aggregate grid is not available yet. See the data preparation instructions.")
    else:
        public_events["year"] = pd.to_numeric(public_events["year"], errors="coerce")
        years = sorted(public_events["year"].dropna().astype(int).unique().tolist())
        states = sorted(public_events["state"].dropna().astype(str).unique().tolist())
        one, two = st.columns(2)
        with one:
            selected_years = st.multiselect("Crash year", years, default=years)
        with two:
            selected_states = st.multiselect("State/UT", states, default=[], help="Leave empty to include every state/UT.")
        subset = public_events[public_events["year"].isin(selected_years)].copy()
        if selected_states:
            subset = subset[subset["state"].astype(str).isin(selected_states)]
        if subset.empty:
            st.info("No records match these filters.")
        else:
            grid = (
                subset.groupby(["cell_latitude", "cell_longitude"], as_index=False)
                .agg(
                    sample_events=("reported_events", "sum"),
                    recorded_fatalities=("recorded_fatalities", "sum"),
                )
            )
            min_count = st.slider("Minimum sampled events per map cell", min_value=5, max_value=20, value=5)
            grid = grid[grid["sample_events"] >= min_count].copy()
            x, y, z = st.columns(3)
            x.metric("Events in published cells", f"{int(grid['sample_events'].sum()):,}")
            y.metric("Recorded fatalities in published cells", f"{int(grid['recorded_fatalities'].sum()):,}")
            z.metric("Published grid cells", f"{len(grid):,}")
            st.caption(
                "Only year × state × 0.25° cells with at least five sampled events are "
                "published. Smaller groups are suppressed before deployment; cell centers are approximate."
            )
            if grid.empty:
                show_missing("No published cell meets the selected support threshold. Widen the filters or lower the threshold.")
            else:
                map_fig = px.scatter_map(
                    grid,
                    lat="cell_latitude",
                    lon="cell_longitude",
                    size="sample_events",
                    color="sample_events",
                    hover_data={
                        "sample_events": ":,",
                        "recorded_fatalities": ":,",
                        "cell_latitude": False,
                        "cell_longitude": False,
                    },
                    zoom=3,
                    center={"lat": 22.8, "lon": 79.0},
                    map_style="open-street-map",
                    title="Coarse-grid concentration of sampled reported events",
                )
                map_fig.update_layout(margin={"r": 0, "t": 45, "l": 0, "b": 0})
                st.plotly_chart(map_fig, width="stretch")
                st.caption("Each symbol represents the approximate centroid of a 0.25° cell.")
            by_state = subset.groupby("state", dropna=True).agg(
                sampled_events=("reported_events", "sum"),
                recorded_fatalities=("recorded_fatalities", "sum"),
            ).reset_index().sort_values("sampled_events", ascending=True).tail(15)
            chart = px.bar(by_state.sort_values("sampled_events"), x="sampled_events", y="state", orientation="h", title="States with most reported events in published cells")
            chart.update_layout(xaxis_title="Events in published cells", yaxis_title="")
            st.plotly_chart(chart, width="stretch")

with tabs[3]:
    st.header("Narrative themes · 2022 articles only")
    nlp = read_json(PROCESSED / "nlp_summary_2022.json")
    themes = read_csv(PROCESSED / "nlp_theme_mentions_2022.csv")
    if not nlp or themes.empty:
        show_missing("The aggregate NLP outputs are not available yet. No article text is loaded in this dashboard.")
    else:
        left, right = st.columns(2)
        left.metric("Unique nonempty articles", metric_or_na(nlp.get("unique_articles_analyzed")))
        right.metric("Articles with missing/unparsed date", metric_or_na(nlp.get("articles_with_unparsed_or_missing_article_date")))
        themes = themes.sort_values("articles_with_mention", ascending=True)
        fig = px.bar(
            themes,
            x="articles_with_mention",
            y="theme",
            orientation="h",
            hover_data={"article_share_pct": ":.2f", "label_type": True},
            title="Articles containing weak keyword themes",
        )
        fig.update_layout(xaxis_title="Unique articles with a mention", yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Themes can overlap. Keyword matches do not resolve negation, attribution, chronology, or causation. "
            "No article body, title, article-level label, or named entity is displayed or exported."
        )

with tabs[4]:
    st.header("Structured ML · conditional fatality-band demonstration")
    ml = read_json(PROCESSED / "conditional_fatality_band_metrics.json")
    if not ml:
        show_missing("The grouped-validation metric output is not available yet.")
    else:
        st.warning(
            "Estimand: P(recorded deaths ≥ 2 | record is already in this media-reported fatal-crash sample). "
            "This is not crash occurrence, causal attribution, or a road-segment hazard score."
        )
        st.caption(f"Rows: {ml.get('rows_used', '—'):,} • validation: {ml.get('groups', {}).get('cv', '—')} • grouped by state")
        model = ml.get("out_of_fold_metrics", {})
        baseline = ml.get("majority_baseline_metrics", {})
        metric_names = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("f1_at_0_5", "F1 at diagnostic 0.5"), ("roc_auc", "ROC-AUC"), ("average_precision", "Average precision"), ("brier_score", "Brier score · lower is better")]
        compare = pd.DataFrame(
            [{"Metric": label, "Grouped-CV model": model.get(key), "Majority baseline": baseline.get(key)} for key, label in metric_names]
        )
        st.dataframe(compare, hide_index=True, width="stretch")
        st.caption(
            "Predictors: road type, crash type, and vehicle categories. State is used only as the grouping unit. "
            "No future-year holdout or external population validation is available."
        )

with tabs[5]:
    st.header("Road-image hazard benchmark · separate CV dataset")
    vision_report = REPORTS / "vision_evaluation.md"
    vision_candidates = [
        PROCESSED / "vision" / "metrics.json",
        PROCESSED / "vision_metrics.json",
        PROCESSED / "vision_summary.json",
    ]
    vision_metrics_path = next((path for path in vision_candidates if path.exists()), None)
    if vision_metrics_path is not None:
        vision = read_json(vision_metrics_path)
        st.write("This evaluation uses the independent RDD2022 India road-image benchmark; its images are not linked to crash records.")
        evaluation = vision.get("evaluation", {}) if isinstance(vision, dict) else {}
        model_info = vision.get("model", {}) if isinstance(vision, dict) else {}
        selected = evaluation.get("selected_checkpoint_metrics", {})

        def cv_metric(value: object) -> str:
            if value is None:
                return "Not defined"
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return "Not available"
            return f"{numeric:.3f}" if math.isfinite(numeric) else "Not defined"

        if selected:
            summary_cols = st.columns(4)
            summary_cols[0].metric("mAP50", cv_metric(selected.get("map50_macro_over_classes_with_ground_truth")))
            summary_cols[1].metric("mAP50:95", cv_metric(selected.get("map50_95_macro_over_classes_with_ground_truth")))
            summary_cols[2].metric("Validation images", metric_or_na(evaluation.get("validation_images_used")))
            epochs = model_info.get("epochs")
            device = model_info.get("device")
            training_summary = f"{device or 'Unknown device'} · {epochs} epoch{'s' if epochs != 1 else ''}" if epochs is not None else f"{device or 'Unknown device'} · epochs n/a"
            summary_cols[3].metric("Training", training_summary)

            class_metrics = selected.get("per_class", {})
            class_rows = []
            for class_name in ("D00", "D10", "D20", "D40"):
                class_result = class_metrics.get(class_name, {}) if isinstance(class_metrics, dict) else {}
                ground_truth = class_result.get("ground_truth_instances")
                ap50 = class_result.get("ap50")
                ap50_95 = class_result.get("ap50_95")
                if ground_truth == 0:
                    ap50_display = "N/A · no validation instances"
                    ap50_95_display = "N/A · no validation instances"
                else:
                    ap50_display = cv_metric(ap50)
                    ap50_95_display = cv_metric(ap50_95)
                class_rows.append(
                    {
                        "Class": class_name,
                        "Validation ground-truth instances": metric_or_na(ground_truth),
                        "Predicted detections": metric_or_na(class_result.get("predicted_instances")),
                        "AP50": ap50_display,
                        "AP50:95": ap50_95_display,
                    }
                )
            st.subheader("Validation results by damage class")
            st.dataframe(pd.DataFrame(class_rows), hide_index=True, width="stretch")

            train_images = evaluation.get("train_images_used")
            validation_images = evaluation.get("validation_images_used")
            full_validation = evaluation.get("full_grouped_validation_images_available")
            train_count, valid_count = st.columns(2)
            train_count.metric("Training images used", metric_or_na(train_images))
            validation_note = f"of {metric_or_na(full_validation)} available" if full_validation is not None else ""
            valid_count.caption(f"Validation images used: {metric_or_na(validation_images)} {validation_note}")
            st.caption(f"Model: {model_info.get('architecture', 'Architecture not recorded')} · checkpoint epoch {evaluation.get('selected_checkpoint_epoch', 'not recorded')}")
            st.warning(
                "The validation split uses sequential filename/image-ID blocks as a sequence proxy; it is not a verified "
                "geographic holdout. The model trained for a compact demonstration run, so these scores are not a "
                "deployment benchmark. RDD images are an independent dataset and are not matched to the crash sample."
            )
            st.caption("AP values are undefined when a class has no ground-truth instances in validation; zero is shown only when the metric was computed as zero.")
        else:
            show_missing("The CV metrics file is present, but its selected-checkpoint aggregate metrics are missing or undefined.")
    elif vision_report.exists():
        st.write("A CV evaluation report is available. The image benchmark is independent of the crash sample.")
        st.markdown(vision_report.read_text(encoding="utf-8"))
    else:
        st.info(
            "Computer-vision outputs have not been generated yet. When available, this section will summarize "
            "the separate road-image benchmark. Its images and detections must not be treated as crash locations."
        )

