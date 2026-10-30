"""Model Benchmarking: every model, tuning, calibration and the final test results."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app\components importable

import json  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from churn.models.imbalance import STRATEGY_LABELS  # noqa: E402
from churn.models.registry import REGISTRY  # noqa: E402
from components.charts import (  # noqa: E402
    confusion_figure,
    metric_heatmap,
    reliability_figure,
    roc_pr_figures,
)
from components.loaders import artifact_path, load_csv, load_json, load_npz  # noqa: E402
from components.ui import (  # noqa: E402
    page_header,
    require_service,
    setup_page,
    show_chart,
    show_df,
    sidebar_model_card,
)

setup_page("Model Benchmarking", "🏁")
service = require_service()
sidebar_model_card(service)
st.sidebar.caption("Thresholds on this page are the stored ones; the slider on other pages "
                   "does not apply to test-set results.")
page_header("Model Benchmarking",
            "From 13 models × 5 imbalance strategies to one deployed model. Every number on "
            "this page comes from saved pipeline outputs.")

LABELS = {key: spec.label for key, spec in REGISTRY.items()}
METRICS = {
    "ROC-AUC": "roc_auc_mean",
    "PR-AUC": "pr_auc_mean",
    "Recall (threshold 0.5)": "recall_mean",
    "Precision (threshold 0.5)": "precision_mean",
    "F1 (threshold 0.5)": "f1_mean",
    "Matthews correlation (MCC)": "mcc_mean",
    "Balanced accuracy": "balanced_accuracy_mean",
    "Brier score (lower is better)": "brier_mean",
    "Training seconds per fold": "fit_seconds_mean",
    "Inference ms per 1,000 rows": "ms_per_1000_rows_mean",
}


def label(model: str) -> str:
    return LABELS.get(model, model)


def run_label(key: str) -> str:
    model, strategy = key.split("__")[:2]
    return f"{label(model)} + {STRATEGY_LABELS.get(strategy, strategy)}"


tab_bench, tab_tune, tab_cal, tab_final = st.tabs(
    ["1 · Benchmark", "2 · Tuning", "3 · Calibration & thresholds", "4 · Final model on the test set"])

# ---------------------------------------------------------------------
with tab_bench:
    summary = load_csv("artifacts/metrics/benchmark/benchmark_summary.csv")
    meta = load_json("artifacts/metrics/benchmark/benchmark_meta.json") or {}
    oof = load_npz("artifacts/metrics/benchmark/benchmark_oof.npz")
    if summary is None:
        st.info("Run: python scripts\\04_benchmark_models.py")
    else:
        st.caption(
            f"{meta.get('runs_ok', '?')} runs, 5-fold stratified cross-validation on the training "
            f"split ({meta.get('rows', '?')} customers, churn {meta.get('churn_rate_pct', '?')}%). "
            f"{meta.get('runs_not_applicable', 0)} combinations were not applicable (no class-weight "
            "support). Precision, recall and F1 use the default 0.5 threshold; ROC-AUC and PR-AUC "
            "do not depend on a threshold."
        )
        metric = st.selectbox("Metric", list(METRICS), key="bench_metric")
        show_chart(metric_heatmap(summary, METRICS[metric], LABELS, STRATEGY_LABELS))
        st.caption("Switch to Recall, then to ROC-AUC: balancing strategies move recall a lot "
                   "but barely change ranking quality. That's the threshold moving, not a "
                   "better model.")

        best = (summary.sort_values("pr_auc_mean", ascending=False)
                .drop_duplicates("model").reset_index(drop=True))
        st.markdown("**Best strategy per model (by PR-AUC)**")
        view = best.assign(model=best["model"].map(label))[
            ["model", "strategy", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "recall_mean",
             "precision_mean", "f1_mean", "fit_seconds_mean", "ms_per_1000_rows_mean"]]
        show_df(view.round(4), hide_index=True)
        st.caption("The std column is fold-to-fold variation; the uncertainty of each mean is "
                   "roughly std ÷ √5. Gaps smaller than that are noise.")

        if oof is not None:
            st.markdown("**ROC and precision-recall curves (out-of-fold predictions)**")
            models = best["model"].tolist()
            chosen = st.multiselect("Models", models, default=models[:5], format_func=label,
                                    key="bench_curves")
            curves = {run_label(f"{m}__{s}"): oof[f"{m}__{s}"]
                      for m, s in zip(best["model"], best["strategy"])
                      if m in chosen and f"{m}__{s}" in oof}
            if curves:
                roc, pr = roc_pr_figures(oof["y_true"], curves)
                left, right = st.columns(2)
                with left:
                    show_chart(roc)
                with right:
                    show_chart(pr)

            st.markdown("**Confusion matrix at the default 0.5 threshold**")
            runs = [f"{m}__{s}" for m, s in zip(summary["model"], summary["strategy"])
                    if f"{m}__{s}__pred" in oof]
            run = st.selectbox("Run", runs, format_func=run_label, key="bench_run")
            show_chart(confusion_figure(oof["y_true"], oof[f"{run}__pred"],
                                        title="Out-of-fold predictions, training split"))

        folds = load_csv("artifacts/metrics/benchmark/benchmark_folds.csv")
        if folds is not None:
            keep = set(zip(best["model"], best["strategy"]))
            mask = [(m, s) in keep for m, s in zip(folds["model"], folds["strategy"])]
            spread = folds[mask].assign(model=lambda d: d["model"].map(label))
            fig = px.box(spread, x="roc_auc", y="model", points="all",
                         labels={"roc_auc": "ROC-AUC per fold", "model": ""},
                         title="Fold-to-fold spread (best strategy per model)")
            fig.update_layout(height=480)
            show_chart(fig)

        fig = px.scatter(best.assign(label=best["model"].map(label)), x="fit_seconds_mean",
                         y="pr_auc_mean", text="label", log_x=True,
                         labels={"fit_seconds_mean": "Training seconds per fold (log scale)",
                                 "pr_auc_mean": "Mean PR-AUC"},
                         title="Speed vs quality")
        fig.update_traces(textposition="top center")
        fig.update_layout(height=440)
        show_chart(fig)

# ---------------------------------------------------------------------
with tab_tune:
    tuning = load_csv("artifacts/metrics/tuning/tuning_summary.csv")
    if tuning is None:
        st.info("Run: python scripts\\05_tune_models.py")
    else:
        st.caption("Stage 2: the five shortlisted models were tuned with 5-fold CV on the training "
                   "split (grid search for logistic regression, randomized search for the others), "
                   "all with strategy 'none'. Delta = tuned minus the Milestone 4 default on the "
                   "same folds.")
        t = tuning.assign(label=tuning["model"].map(label))
        values = pd.concat([t["baseline_pr_auc"], t["cv_pr_auc"]]).dropna()
        fig = go.Figure([
            go.Bar(name="Default (Milestone 4)", y=t["label"], x=t["baseline_pr_auc"],
                   orientation="h", marker_color="#B0B0B0"),
            go.Bar(name="Tuned", y=t["label"], x=t["cv_pr_auc"], orientation="h",
                   error_x=dict(type="data", array=t["cv_pr_auc_std"]),
                   marker_color="#1F4E79"),
        ])
        fig.update_layout(barmode="group", xaxis_title="Mean CV PR-AUC", height=400,
                          xaxis_range=[max(0.0, float(values.min()) - 0.05),
                                       float(values.max()) + 0.05])
        show_chart(fig)
        columns = ["label", "search_type", "n_candidates", "cv_pr_auc", "cv_pr_auc_std",
                   "baseline_pr_auc", "delta_pr_auc", "cv_roc_auc", "baseline_roc_auc",
                   "delta_roc_auc", "overfit_gap", "search_seconds"]
        show_df(t[[c for c in columns if c in t.columns]].round(4), hide_index=True)
        st.caption("overfit_gap = training PR-AUC minus CV PR-AUC of the chosen settings. The best "
                   "of many candidates scored on the same folds is slightly optimistic.")
        for row in t.itertuples(index=False):
            with st.expander(f"Best parameters: {row.label}"):
                st.json(json.loads(row.best_params))

# ---------------------------------------------------------------------
with tab_cal:
    comparison = load_csv("artifacts/metrics/calibration/validation_comparison.csv")
    probs = load_npz("artifacts/metrics/calibration/validation_probabilities.npz")
    if comparison is None:
        st.info("Run: python scripts\\06_calibrate_and_threshold.py")
    else:
        st.caption("Every tuned model fitted on the training split and scored once on the "
                   "validation split: uncalibrated, and with sigmoid or isotonic calibration "
                   "(fitted inside the training split).")
        view = comparison.assign(model=comparison["model"].map(label))[
            ["model", "calibration", "roc_auc", "pr_auc", "brier", "log_loss", "ece",
             "fit_seconds"]]
        show_df(view.round(4), hide_index=True)
        st.caption("roc_auc / pr_auc: ranking (higher is better). brier / log_loss / ece: "
                   "probability quality (lower is better).")
        if probs is not None:
            keys = [k for k in probs if k != "y_val"]
            final_model = service.metadata.get("model", "")
            default = [k for k in keys if k.startswith(f"{final_model}__")] or keys[:3]
            chosen = st.multiselect("Calibration curves for", keys, default=default,
                                    format_func=lambda k: f"{label(k.split('__')[0])} "
                                                          f"({k.split('__')[1]})",
                                    key="cal_keys")
            if chosen:
                curves = {f"{label(k.split('__')[0])} ({k.split('__')[1]})": probs[k]
                          for k in chosen}
                show_chart(reliability_figure(probs["y_val"], curves))

        curves_table = load_csv("artifacts/metrics/calibration/learning_curves.csv")
        if curves_table is not None:
            long = pd.concat([
                curves_table.assign(curve="Training", score=curves_table["train_pr_auc"],
                                    spread=curves_table["train_std"]),
                curves_table.assign(curve="Cross-validation", score=curves_table["cv_pr_auc"],
                                    spread=curves_table["cv_std"]),
            ], ignore_index=True)
            long["model"] = long["model"].map(label)
            fig = px.line(long, x="train_rows", y="score", color="curve", facet_col="model",
                          facet_col_wrap=3, markers=True, error_y="spread",
                          labels={"train_rows": "Training rows", "score": "PR-AUC"},
                          title="Learning curves")
            fig.update_layout(height=620, legend_title_text="")
            show_chart(fig)
            st.caption("A large gap between the curves means the model memorizes (variance). "
                       "A cross-validation curve still rising at the right means more data would help.")

        choices = load_csv("artifacts/metrics/calibration/threshold_choices.csv")
        if choices is not None:
            picked = choices[(choices["model"] == service.metadata.get("model"))
                             & (choices["calibration"] == service.metadata.get("calibration"))]
            if not picked.empty:
                st.markdown("**Threshold policies for the deployed model (validation split)**")
                show_df(picked[["policy", "threshold", "precision", "recall", "f1", "flagged_pct",
                                "net_benefit_per_1000"]].round(3), hide_index=True)
                st.caption("Net benefit uses PLACEHOLDER business costs.")

# ---------------------------------------------------------------------
with tab_final:
    meta = service.metadata
    threshold_info = meta.get("threshold", {})
    deployed = float(threshold_info.get("value", 0.5))
    st.markdown(f"**Deployed model:** {meta.get('model_label', '')} ({meta.get('calibration', '')}), "
                f"threshold {deployed:.2f} ({threshold_info.get('policy', '')}). "
                f"Selected by {meta.get('selected_by', '')}: {meta.get('selection_reason', '')}.")
    with st.expander("The selection rule (fixed before the test split was opened)"):
        for line in meta.get("selection_rule", []):
            st.markdown(f"- {line}")
    evidence = load_csv("artifacts/metrics/final/selection_evidence.csv")
    if evidence is not None:
        st.markdown("**Validation evidence behind the choice**")
        columns = ["model", "calibration", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high",
                   "gap_to_reference", "gap_ci_low", "gap_ci_high", "brier", "eligible"]
        show_df(evidence[[c for c in columns if c in evidence.columns]].round(4), hide_index=True)

    test = load_json("artifacts/metrics/final/test_metrics.json")
    if test is None:
        st.info("Run: python scripts\\07_finalize_model.py --evaluate-test")
    else:
        ci = test["confidence_intervals"]
        cols = st.columns(3)
        for col, (name, key) in zip(cols, [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"),
                                           ("Brier score", "brier")]):
            col.metric(f"Test {name}", f"{ci[key]['value']:.4f}",
                       help=f"95% bootstrap interval {ci[key]['ci_low']:.4f} to "
                            f"{ci[key]['ci_high']:.4f}")
        st.caption(f"{test['n_rows']:,} test customers, churn rate {test['churn_rate']:.1%}. "
                   f"Evaluated {meta.get('test_evaluation_count', 1)} time(s), after the model "
                   "and threshold were fixed.")
        policies = pd.DataFrame(test["policies"]).T.reset_index().rename(columns={"index": "policy"})
        columns = ["policy", "threshold", "precision", "recall", "f1", "specificity",
                   "balanced_accuracy", "mcc", "accuracy", "flagged_pct", "net_benefit_per_1000"]
        show_df(policies[[c for c in columns if c in policies.columns]].round(3), hide_index=True)

        chosen_metrics = test["policies"][test["chosen_policy"]]
        tp, fp, fn = chosen_metrics["tp"], chosen_metrics["fp"], chosen_metrics["fn"]
        st.markdown(
            f"**What the errors mean for the retention team.** At threshold {deployed:.2f} the model "
            f"caught **{tp}** of {tp + fn} churners in the test set (recall "
            f"{chosen_metrics['recall']:.1%}) and raised **{fp}** false alarms. A **false negative** "
            "is a churner nobody contacts: the revenue of a customer who leaves. A **false "
            "positive** is a loyal customer who gets an offer: the cost of the incentive and the "
            "contact time. Lowering the threshold catches more churners but adds false alarms; "
            "explore that trade-off on the Risk Analysis page."
        )

    test_probs = load_npz("artifacts/metrics/final/test_probabilities.npz")
    if test_probs is not None:
        y_test, p_test = test_probs["y_true"], test_probs["probability"]
        roc, pr = roc_pr_figures(y_test, {"Final model (test)": p_test}, marker_threshold=deployed)
        left, right = st.columns(2)
        with left:
            show_chart(roc)
        with right:
            show_chart(pr)
        left, right = st.columns(2)
        with left:
            show_chart(reliability_figure(y_test, {"Final model (test)": p_test}))
        with right:
            show_chart(confusion_figure(y_test, (p_test >= deployed).astype(int),
                                        title=f"Test split, threshold {deployed:.2f}"))
    report_file = artifact_path("artifacts/metrics/final/test_classification_report.txt")
    if report_file.exists():
        st.markdown("**Classification report (test split, deployed threshold)**")
        st.code(report_file.read_text(encoding="utf-8"))
    st.caption("Choosing a threshold by looking at test results and then quoting the test score "
               "would leak information, so this tab only shows the stored thresholds.")