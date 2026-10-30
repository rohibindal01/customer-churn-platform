"""Plotly figure builders shared by the app pages.

Pure functions: data in, figure out. No Streamlit calls, so they are easy to test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

CLASS_NAMES = ["Retained", "Churned"]
RAISES, LOWERS = "#C0392B", "#2E86C1"


def roc_pr_figures(y_true, curves: dict, marker_threshold: float | None = None):
    """ROC and precision-recall figures for one or more score arrays.

    With marker_threshold, the first curve gets a marker at that operating point.
    """
    y = np.asarray(y_true).astype(int)
    roc, pr = go.Figure(), go.Figure()
    for name, scores in curves.items():
        s = np.asarray(scores, dtype=float)
        fpr, tpr, _ = roc_curve(y, s)
        roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"{name} (AUC {auc(fpr, tpr):.3f})"))
        precision, recall, _ = precision_recall_curve(y, s)
        pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                                name=f"{name} (AP {average_precision_score(y, s):.3f})"))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(dash="dash", color="grey")))
    pr.add_hline(y=float(y.mean()), line_dash="dash", line_color="grey")

    if marker_threshold is not None and curves:
        s = np.asarray(next(iter(curves.values())), dtype=float)
        flagged = s >= marker_threshold
        tp = int((flagged & (y == 1)).sum())
        fp = int((flagged & (y == 0)).sum())
        fn = int((~flagged & (y == 1)).sum())
        tn = int((~flagged & (y == 0)).sum())
        recall = tp / (tp + fn) if tp + fn else 0.0
        label = f"Threshold {marker_threshold:.2f}"
        roc.add_trace(go.Scatter(x=[fp / (fp + tn) if fp + tn else 0.0], y=[recall],
                                 mode="markers", name=label,
                                 marker=dict(size=12, color=RAISES)))
        pr.add_trace(go.Scatter(x=[recall], y=[tp / (tp + fp) if tp + fp else 0.0],
                                mode="markers", name=label,
                                marker=dict(size=12, color=RAISES)))

    roc.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate (recall)",
                      title="ROC curve", height=420, legend=dict(font=dict(size=10)))
    pr.update_layout(xaxis_title="Recall", yaxis_title="Precision",
                     title="Precision-recall curve (dashed = churn rate)", height=420,
                     legend=dict(font=dict(size=10)))
    return roc, pr


def reliability_figure(y_true, probabilities: dict, n_bins: int = 10) -> go.Figure:
    """Calibration curves: predicted probability vs observed churn rate."""
    y = np.asarray(y_true).astype(int)
    fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                                line=dict(dash="dash", color="grey"))])
    for name, p in probabilities.items():
        observed, predicted = calibration_curve(y, np.asarray(p, dtype=float), n_bins=n_bins,
                                                strategy="quantile")
        fig.add_trace(go.Scatter(x=predicted, y=observed, mode="lines+markers", name=name))
    fig.update_layout(xaxis_title="Mean predicted probability", yaxis_title="Observed churn rate",
                      xaxis_range=[0, 1], yaxis_range=[0, 1], height=420,
                      title="Calibration curve")
    return fig


def confusion_figure(y_true, y_pred, title: str = "") -> go.Figure:
    """Confusion matrix (rows = actual, columns = predicted)."""
    cm = confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int),
                          labels=[0, 1])
    fig = px.imshow(cm, x=[f"Predicted {c}" for c in CLASS_NAMES],
                    y=[f"Actual {c}" for c in CLASS_NAMES], text_auto=True,
                    color_continuous_scale="Blues")
    fig.update_layout(title=title, height=340, coloraxis_showscale=False)
    return fig


def metric_heatmap(summary: pd.DataFrame, metric: str, model_labels: dict,
                   strategy_labels: dict) -> go.Figure:
    """Model x strategy heatmap of one benchmark metric."""
    pivot = summary.pivot(index="model", columns="strategy", values=metric)
    rows = [m for m in model_labels if m in pivot.index]
    cols = [s for s in strategy_labels if s in pivot.columns]
    pivot = pivot.loc[rows, cols]
    pivot.index = [model_labels[m] for m in pivot.index]
    pivot.columns = [strategy_labels[s] for s in pivot.columns]
    fig = px.imshow(pivot.round(4), text_auto=".3f", aspect="auto",
                    color_continuous_scale="Viridis", labels={"color": metric})
    fig.update_layout(height=60 + 34 * len(rows), xaxis_title="", yaxis_title="")
    return fig


def shap_beeswarm(shap_values, X_t, feature_names, top_n: int = 15, max_points: int = 1500,
                  seed: int = 0) -> go.Figure:
    """One row per feature, one dot per customer; colour = feature value (low to high)."""
    shap_values = np.asarray(shap_values, dtype=float)
    X_t = np.asarray(X_t, dtype=float)
    names = list(feature_names)
    order = np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:top_n]
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(shap_values), size=min(max_points, len(shap_values)), replace=False)
    frames = []
    for rank, j in enumerate(order):
        values = X_t[rows, j]
        lo, hi = np.percentile(values, [5, 95])
        colour = np.clip((values - lo) / (hi - lo), 0, 1) if hi > lo else np.full(len(rows), 0.5)
        frames.append(pd.DataFrame({
            "feature": names[j], "shap": shap_values[rows, j],
            "row": rank + rng.uniform(-0.3, 0.3, len(rows)), "value": colour,
        }))
    data = pd.concat(frames, ignore_index=True)
    fig = px.scatter(data, x="shap", y="row", color="value", color_continuous_scale="RdBu_r",
                     range_color=[0, 1], opacity=0.6,
                     hover_data={"feature": True, "row": False, "value": ":.2f"})
    fig.update_traces(marker=dict(size=4))
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=[names[j] for j in order],
                     autorange="reversed", title="")
    fig.add_vline(x=0, line_color="grey")
    fig.update_layout(xaxis_title="SHAP value (log-odds); right = higher churn risk",
                      coloraxis_colorbar=dict(title="Feature value", tickvals=[0, 1],
                                              ticktext=["low", "high"]),
                      height=120 + 32 * len(order))
    return fig


def signed_bar(labels, values, x_title: str, height: int | None = None) -> go.Figure:
    """Horizontal bars: red raises churn risk, blue lowers it. First item at the top."""
    values = [float(v) for v in values]
    fig = go.Figure(go.Bar(x=values, y=list(labels), orientation="h",
                           marker_color=[RAISES if v > 0 else LOWERS for v in values]))
    fig.update_yaxes(autorange="reversed")
    fig.add_vline(x=0, line_color="grey")
    fig.update_layout(xaxis_title=x_title, height=height or (120 + 30 * len(values)),
                      showlegend=False)
    return fig