"""Leakage-controlled descriptive severity model for Mendeley fatal-crash rows.

The target is whether a *reported fatal crash* has two or more recorded deaths.
It is not a crash-occurrence, road-risk, causal, or intervention model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data/processed/crash_events_clean.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed"
DEFAULT_REPORT = PROJECT_ROOT / "reports/nlp_and_model_evaluation.md"
RANDOM_SEED = 42
FEATURES = ["road_type", "crash_type", "vehicle_1", "vehicle_2"]
TARGET_DEFINITION = "1 if Killed >= 2, otherwise 0 (one recorded death)"


def _metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    predicted = (probability >= threshold).astype(int)
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision_at_0_5": float(precision_score(y_true, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, predicted, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }
    out["roc_auc"] = float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None
    out["average_precision"] = float(average_precision_score(y_true, probability)) if len(np.unique(y_true)) == 2 else None
    out["positive_prevalence"] = float(np.mean(y_true)) if len(y_true) else None
    return out


def _pipeline(class_weight: str | None = "balanced") -> Pipeline:
    prep = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value="Not stated")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
                ]),
                FEATURES,
            ),
        ],
        remainder="drop",
    )
    model = LogisticRegression(
        max_iter=2000,
        class_weight=class_weight,
        solver="liblinear",
        random_state=RANDOM_SEED,
    )
    return Pipeline([("features", prep), ("classifier", model)])


def _cluster_bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
    metric_name: str,
    iterations: int = 500,
) -> dict[str, float] | None:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 5:
        return None
    rng = np.random.default_rng(RANDOM_SEED)
    scores: list[float] = []
    rows_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
    for _ in range(iterations):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([rows_by_group[g] for g in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        if metric_name == "f1":
            scores.append(float(f1_score(y[idx], pred[idx], zero_division=0)))
        elif metric_name == "average_precision":
            scores.append(float(average_precision_score(y[idx], pred[idx])))
        elif metric_name == "roc_auc":
            scores.append(float(roc_auc_score(y[idx], pred[idx])))
    if not scores:
        return None
    return {"lower_95": float(np.quantile(scores, 0.025)), "upper_95": float(np.quantile(scores, 0.975)), "replicates": len(scores)}


def evaluate_model(input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Clean events not found at {input_path}. Run `python -m road_safety.crash_data` first."
        )
    events = pd.read_csv(input_path)
    required = set(FEATURES + ["killed", "state", "crash_date"])
    if not required.issubset(events.columns):
        raise ValueError(f"Clean event file is missing model columns: {sorted(required.difference(events.columns))}")
    events["killed"] = pd.to_numeric(events["killed"], errors="coerce")
    events = events.loc[events["killed"].notna() & events["state"].notna() & events["killed"].ge(1)].copy()
    if events.empty:
        raise ValueError("No fatal-crash rows with a valid state and Killed value remain.")
    y = events["killed"].ge(2).astype(int).to_numpy()
    groups = events["state"].astype(str).to_numpy()
    X = events[FEATURES].copy()
    if len(np.unique(y)) < 2:
        raise ValueError("The selected fatality-band target has only one class; do not fit a classifier.")

    n_groups = int(events["state"].nunique())
    n_splits = min(5, n_groups)
    if n_splits < 3:
        raise ValueError(f"At least three independent state groups are needed; found {n_groups}.")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    oof_prob = np.full(len(events), np.nan, dtype=float)
    base_prob = np.full(len(events), np.nan, dtype=float)
    fold_results = []
    fold_assignments = np.full(len(events), -1, dtype=int)

    for fold_num, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups), start=1):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[valid_idx])) < 2:
            raise ValueError(
                "A state-group fold lacks one target class. Revise the group split only after inspecting "
                "the source geography and report the resulting validation limit."
            )
        model = _pipeline(class_weight="balanced")
        model.fit(X.iloc[train_idx], y[train_idx])
        oof_prob[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
        baseline = DummyClassifier(strategy="most_frequent").fit(X.iloc[train_idx], y[train_idx])
        base_prob[valid_idx] = baseline.predict_proba(X.iloc[valid_idx])[:, 1]
        fold_assignments[valid_idx] = fold_num
        fold_metrics = _metrics(y[valid_idx], oof_prob[valid_idx])
        baseline_metrics = _metrics(y[valid_idx], base_prob[valid_idx])
        fold_results.append({
            "fold": fold_num,
            "held_out_states": sorted(set(groups[valid_idx])),
            "rows": int(len(valid_idx)),
            "model": fold_metrics,
            "majority_baseline": baseline_metrics,
        })

    if np.isnan(oof_prob).any() or np.isnan(base_prob).any():
        raise RuntimeError("Grouped cross-validation did not generate one out-of-fold prediction per event.")
    model_metrics = _metrics(y, oof_prob)
    baseline_metrics = _metrics(y, base_prob)
    model_metrics["cluster_bootstrap_ci"] = {
        metric: _cluster_bootstrap_ci(y, (oof_prob >= 0.5).astype(int) if metric == "f1" else oof_prob, groups, metric)
        for metric in ("f1", "average_precision", "roc_auc")
    }
    # Fit the deployable artifact on all available data only after reporting
    # grouped out-of-fold evaluation; the artifact is not a calibrated risk score.
    fitted_model = _pipeline(class_weight="balanced").fit(X, y)
    output_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(fitted_model, output_dir / "conditional_fatality_band_logistic.joblib")

    metrics: dict[str, Any] = {
        "project": "India Road-Crash Hotspot and Hazard Analysis",
        "source": "Mendeley Data v7 structured workbook; fatal crashes reported by Times of India",
        "estimand": "P(recorded Killed >= 2 | event is in this source's reported fatal-crash sample, four structured event features)",
        "target_definition": TARGET_DEFINITION,
        "rows_used": int(len(events)),
        "class_counts": {"one_recorded_death": int(np.sum(y == 0)), "two_or_more_recorded_deaths": int(np.sum(y == 1))},
        "positive_prevalence": float(np.mean(y)),
        "groups": {"field": "State", "count": n_groups, "cv": f"{n_splits}-fold StratifiedGroupKFold; whole states held out"},
        "features": FEATURES,
        "excluded_fields": [
            "Killed except as target", "Injured (post-event and outcome-correlated)", "Age", "Gender",
            "Article Date (publication/reporting leakage)", "State (used only for group separation)",
            "Location, coordinates, row number, and source narrative",
        ],
        "validation": {
            "strategy": "Pooled out-of-fold predictions from stratified grouped CV by state; no state appears in both train and validation within a fold.",
            "temporal_holdout": False,
            "split_unit": "State",
            "threshold": 0.5,
            "threshold_note": "Diagnostic default only; not an operational decision threshold.",
            "seed": RANDOM_SEED,
        },
        "model": {"type": "one-hot encoded logistic regression", "class_weight": "balanced", "imputer": "constant Not stated", "rare_category_min_frequency": 2},
        "out_of_fold_metrics": model_metrics,
        "majority_baseline_metrics": baseline_metrics,
        "folds": fold_results,
        "model_artifact": "data/processed/conditional_fatality_band_logistic.joblib",
        "limitations": [
            "All records are already fatal crashes; model cannot predict crash occurrence or a road segment's future crash rate.",
            "Events are media-selected and the sample is not representative of all police-reported or unreported crashes.",
            "No traffic exposure, complete nonfatal denominator, or linked road-segment panel is available.",
            "Grouped validation tests transfer to held-out states within this source, not future-year performance.",
            "Probabilities are exploratory, not calibrated for emergency, enforcement, engineering, insurance, or public decisions.",
            "Associations are predictive descriptions, not causal effects or evidence that an attribute caused the crash.",
        ],
    }
    (output_dir / "conditional_fatality_band_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_evaluation_report(metrics, DEFAULT_REPORT)
    return metrics


def write_evaluation_report(metrics: dict[str, Any], path: Path = DEFAULT_REPORT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nlp_path = PROJECT_ROOT / "data/processed/nlp_summary_2022.json"
    nlp: dict[str, Any] = json.loads(nlp_path.read_text(encoding="utf-8")) if nlp_path.exists() else {}
    model = metrics["out_of_fold_metrics"]
    baseline = metrics["majority_baseline_metrics"]
    confusion = model["confusion_matrix_labels_0_1"]
    theme_rows = []
    for theme, record in sorted(nlp.get("theme_mentions", {}).items(), key=lambda kv: kv[1]["articles"], reverse=True):
        theme_rows.append(f"| `{theme}` | {record['articles']:,} | {record['share_pct']:.2f}% |")
    if not theme_rows:
        theme_rows = ["| No raw narrative artifact found | — | — |"]
    fold_rows = []
    for fold in metrics["folds"]:
        fm = fold["model"]
        fold_rows.append(
            f"| {fold['fold']} | {fold['rows']:,} | {len(fold['held_out_states'])} | "
            f"{fm['f1_at_0_5']:.3f} | {fm['roc_auc']:.3f} | {fm['average_precision']:.3f} |"
        )
    if not fold_rows:
        fold_rows = ["| — | — | — | — | — | — |"]
    report = f"""# NLP and structured model evaluation

## Narrative NLP: 2022 raw article corpus

- **Coverage:** {nlp.get('unique_articles_analyzed', 0):,} unique nonempty articles from the local 2022 raw CSV, after removing {nlp.get('exact_duplicate_articles_removed_for_counts', 0):,} exact duplicate article bodies for aggregate counts.
- **Observed columns:** title `{nlp.get('title_column_detected', 'not available')}`, date `{nlp.get('article_date_column_detected', 'not available')}`, content `{nlp.get('content_column_detected', 'not available')}`.
- **Article-date range:** {nlp.get('article_date_min', 'not available')} to {nlp.get('article_date_max', 'not available')}.
- **Coverage limitation:** only the 2022 raw file was available. The attempted 2023 raw download was a 395-byte internal-error JSON response and was rejected as data.

We ran transparent dictionary-pattern extraction across title and content. The counts below are **articles containing a keyword theme**, not verified crash factors. Themes can overlap; negation, attribution, and temporal sequence are not resolved. Article-level matches and text are not exported. No hand-labeled gold sample was available, so precision, recall, F1, and NLP model accuracy are intentionally not reported.

| Weak mention theme | Articles | Share of unique 2022 articles |
|---|---:|---:|
{chr(10).join(theme_rows)}

These counts describe coverage of a newspaper corpus of fatal-crash stories. They cannot estimate prevalence among all crashes or prove that a mentioned factor caused a crash. For valid NLP evaluation, annotate an event-grouped sample with a written codebook, dual-label a subset, report agreement, freeze an article/event-grouped holdout, and score entity/attribute extraction against human gold labels.

Raw titles and article bodies remain under `data/raw/mendeley/v7/raw_articles/` locally. Processed NLP outputs contain only theme-level counts and source metadata. Do not publish raw article text or named entities.

## Structured ML: conditional fatality-band demonstration

**Question actually modeled:** among this dataset's media-reported fatal crashes, can a small set of source-coded roadway/collision/vehicle attributes distinguish rows with one recorded death from those with two or more recorded deaths?

The dataset has {metrics['rows_used']:,} usable rows: {metrics['class_counts']['one_recorded_death']:,} with one death and {metrics['class_counts']['two_or_more_recorded_deaths']:,} with two or more. Positive class prevalence is {metrics['positive_prevalence']:.1%}. The binary cutoff is a simple, predeclared demonstration label, not an official severity standard.

### Validation design

- Five-fold (or fewer if there are fewer state groups) stratified grouped cross-validation; complete states are held out together. The reported pooled metrics are from out-of-fold predictions.
- Fixed logistic regression with one-hot categorical encoding, rare-category minimum frequency 2, and balanced class weights. No hyperparameter search was run.
- Predictors: `{', '.join(metrics['features'])}`.
- `State` is used only to keep geographic groups apart; it is not a predictor. `Injured`, article publication date, age/gender, coordinates/location, serial number, and narrative text are excluded.
- A majority-class `DummyClassifier` is the baseline. The 0.5 cutoff is diagnostic only; it is not an action threshold.

| Metric | Grouped OOF model | Majority baseline |
|---|---:|---:|
| Accuracy | {model['accuracy']:.3f} | {baseline['accuracy']:.3f} |
| Balanced accuracy | {model['balanced_accuracy']:.3f} | {baseline['balanced_accuracy']:.3f} |
| Precision (0.5) | {model['precision_at_0_5']:.3f} | {baseline['precision_at_0_5']:.3f} |
| Recall (0.5) | {model['recall_at_0_5']:.3f} | {baseline['recall_at_0_5']:.3f} |
| F1 (0.5) | {model['f1_at_0_5']:.3f} | {baseline['f1_at_0_5']:.3f} |
| ROC-AUC | {model['roc_auc']:.3f} | {baseline['roc_auc']:.3f} |
| Average precision | {model['average_precision']:.3f} | {baseline['average_precision']:.3f} |
| Brier score (lower better) | {model['brier_score']:.3f} | {baseline['brier_score']:.3f} |

Pooled model confusion matrix with rows=true `[one death, 2+ deaths]` and columns=predicted `[one death, 2+ deaths]`: `{confusion}`.

### Fold stability

| Fold | Held-out rows | Held-out states | F1 | ROC-AUC | Average precision |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(fold_rows)}

Cluster bootstrap intervals in `data/processed/conditional_fatality_band_metrics.json` resample state groups from pooled out-of-fold predictions; they measure sample instability, not population coverage or external validity.

## Model card and use limits

| Item | Description |
|---|---|
| Model | One-hot encoded, class-weighted logistic regression; fitted artifact saved at `data/processed/conditional_fatality_band_logistic.joblib`. |
| Intended use | Reproducible portfolio demonstration of structured modeling and grouped validation among the source's already-fatal reported events. |
| Prohibited interpretation | Do not use as a road hotspot, crash occurrence, individual, causal, emergency-response, enforcement, insurance, infrastructure funding, or engineering action score. |
| Training/evaluation data | Mendeley v7 Times of India media-reported fatal crashes; English newspaper source; observed period {nlp.get('article_date_min', '2022 records')}–{nlp.get('article_date_max', '2023 records')} for articles if available. Structured event range is in `reports/data_quality.md`. |
| Target | `Killed >= 2`, conditional on inclusion in this fatal-crash media sample. |
| Validation | Grouped by state; no future-year or external-source validation. |
| Known bias | Fatal-only selection, newsworthiness/editorial coverage, location/source representation, extracted-field error, sparse categories, and no all-crash or traffic-volume denominator. |
| Calibration | Brier score is reported, but probabilities are not calibrated for decisions; do not set operational cutoffs. |
| Monitoring needed before any future authorized use | Source/schema drift, state and road-class coverage, target definitions, missingness, groupwise calibration and error, independent representative labels, and traffic exposure. |

**Conclusion:** this experiment can demonstrate preprocessing, a leakage-aware target, group holdout, baseline comparison, and honest reporting. It cannot establish which Indian road segments are dangerous. A defensible hotspot/risk model needs representative police/health crash records linked to a road network plus traffic exposure (for example vehicle-kilometres travelled), followed by spatial and future-period validation.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = evaluate_model(args.input, args.output_dir)
    print(json.dumps({
        "rows_used": result["rows_used"],
        "class_counts": result["class_counts"],
        "grouped_oof_metrics": result["out_of_fold_metrics"],
        "majority_baseline": result["majority_baseline_metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
