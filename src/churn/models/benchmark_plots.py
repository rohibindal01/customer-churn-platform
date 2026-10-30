"""Figures for the benchmark results."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from churn.models.benchmark import BenchmarkResult, best_per_model  # noqa: E402
from churn.models.evaluation import CLASS_NAMES  # noqa: E402
from churn.models.imbalance import STRATEGIES, STRATEGY_LABELS  # noqa: E402
from churn.models.registry import REGISTRY, list_model_keys  # noqa: E402


def _label(model_key: str) -> str:
    return REGISTRY[model_key].label if model_key in REGISTRY else model_key


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metric_heatmap(summary, metric: str, title: str, path: Path) -> Path:
    pivot = summary.pivot(index="model", columns="strategy", values=metric)
    rows = [m for m in list_model_keys() if m in pivot.index]
    cols = [s for s in STRATEGIES if s in pivot.columns]
    pivot = pivot.loc[rows, cols]
    pivot.index = [_label(m) for m in pivot.index]
    pivot.columns = [STRATEGY_LABELS[s] for s in pivot.columns]
    fig, ax = plt.subplots(figsize=(1.7 * len(cols) + 3, 0.55 * len(rows) + 2))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    return _save(fig, path)


def plot_roc_pr_curves(result: BenchmarkResult, best, roc_path: Path, pr_path: Path):
    y = result.y_true
    cmap = plt.get_cmap("tab20")
    fig_roc, ax_roc = plt.subplots(figsize=(8.5, 6.5))
    fig_pr, ax_pr = plt.subplots(figsize=(8.5, 6.5))
    for i, row in enumerate(best.itertuples(index=False)):
        scores = result.oof_scores.get(f"{row.model}__{row.strategy}")
        if scores is None or np.isnan(scores).any():
            continue
        name = f"{_label(row.model)} ({row.strategy})"
        fpr, tpr, _ = roc_curve(y, scores)
        precision, recall, _ = precision_recall_curve(y, scores)
        ax_roc.plot(fpr, tpr, color=cmap(i % 20), label=f"{name} AUC={auc(fpr, tpr):.3f}")
        ax_pr.plot(recall, precision, color=cmap(i % 20),
                   label=f"{name} AP={average_precision_score(y, scores):.3f}")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curves (out-of-fold, best strategy per model)")
    ax_roc.legend(fontsize=7, loc="lower right")
    ax_pr.axhline(float(y.mean()), color="k", linestyle="--", linewidth=0.8)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-recall curves (dashed line = churn rate)")
    ax_pr.legend(fontsize=7, loc="upper right")
    return _save(fig_roc, roc_path), _save(fig_pr, pr_path)


def plot_time_bars(best, path: Path) -> Path:
    labels = [_label(m) for m in best["model"]]
    fig, (ax_fit, ax_score) = plt.subplots(1, 2, figsize=(12, 0.45 * len(labels) + 2))
    ax_fit.barh(labels, best["fit_seconds_mean"], color="#4C72B0")
    ax_fit.set_xlabel("Mean training time per fold (seconds)")
    ax_fit.invert_yaxis()
    ax_score.barh(labels, best["ms_per_1000_rows_mean"], color="#55A868")
    ax_score.set_xlabel("Inference time per 1,000 rows (milliseconds, incl. preprocessing)")
    ax_score.invert_yaxis()
    ax_score.set_yticklabels([])
    return _save(fig, path)


def plot_confusion_grid(result: BenchmarkResult, best, path: Path) -> Path:
    n_cols = 4
    n_rows = int(np.ceil(len(best) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.7 * n_cols, 3.5 * n_rows), squeeze=False)
    flat = axes.ravel()
    for ax, row in zip(flat, best.itertuples(index=False)):
        pred = result.oof_pred[f"{row.model}__{row.strategy}"]
        cm = confusion_matrix(result.y_true, pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], CLASS_NAMES, fontsize=7)
        ax.set_yticks([0, 1], CLASS_NAMES, fontsize=7)
        ax.set_title(f"{_label(row.model)}\n({row.strategy}, default threshold)", fontsize=8)
    for ax in flat[len(best):]:
        ax.set_visible(False)
    fig.suptitle("Out-of-fold confusion matrices (rows = actual, columns = predicted)", y=1.0)
    return _save(fig, path)


def make_all_plots(result: BenchmarkResult, out_dir: Path | str) -> list[Path]:
    """Create every benchmark figure and return the file paths."""
    if result.summary.empty:
        return []
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    summary = result.summary
    best = best_per_model(summary, "pr_auc_mean")
    files = [
        plot_metric_heatmap(summary, "roc_auc_mean", "Mean ROC-AUC by model and strategy",
                            directory / "heatmap_roc_auc.png"),
        plot_metric_heatmap(summary, "pr_auc_mean", "Mean PR-AUC by model and strategy",
                            directory / "heatmap_pr_auc.png"),
        plot_metric_heatmap(summary, "recall_mean", "Mean churn recall at default threshold",
                            directory / "heatmap_recall.png"),
    ]
    files.extend(plot_roc_pr_curves(result, best, directory / "roc_curves.png",
                                    directory / "pr_curves.png"))
    files.append(plot_time_bars(best, directory / "timing.png"))
    files.append(plot_confusion_grid(result, best, directory / "confusion_matrices.png"))
    return files