"""Figures for calibration, thresholds and learning curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

from churn.config.settings import BusinessSettings  # noqa: E402
from churn.models.registry import REGISTRY  # noqa: E402
from churn.models.threshold import breakeven_probability  # noqa: E402


def _label(model_key: str) -> str:
    return REGISTRY[model_key].label if model_key in REGISTRY else model_key


def _grid(n_panels: int, width: float = 5.0, height: float = 4.3):
    n_cols = min(3, max(n_panels, 1))
    n_rows = int(np.ceil(max(n_panels, 1) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width * n_cols, height * n_rows),
                             squeeze=False)
    flat = axes.ravel()
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    return fig, flat


def _save(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_reliability(results, y_val, path: Path | str, n_bins: int = 10) -> Path:
    """One calibration curve panel per model, one line per calibration method."""
    ok = [r for r in results if r.status == "ok"]
    models = list(dict.fromkeys(r.model for r in ok))
    fig, flat = _grid(len(models))
    for ax, model in zip(flat, models):
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
        for r in ok:
            if r.model != model:
                continue
            observed, predicted = calibration_curve(y_val, r.val_prob, n_bins=n_bins,
                                                    strategy="quantile")
            ax.plot(predicted, observed, marker="o", markersize=3,
                    label=f"{r.calibration} (Brier {r.metrics['brier']:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(_label(model))
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed churn rate")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Calibration curves on the validation split", y=1.0)
    return _save(fig, path)


def plot_threshold_curves(sweeps: dict, business: BusinessSettings, path: Path | str) -> Path:
    """Precision, recall, F1 and net benefit against the threshold, one panel per model."""
    fig, flat = _grid(len(sweeps), width=5.5)
    breakeven = breakeven_probability(business)
    for ax, (label, sweep) in zip(flat, sweeps.items()):
        t = sweep["threshold"]
        ax.plot(t, sweep["precision"], label="Precision")
        ax.plot(t, sweep["recall"], label="Recall")
        ax.plot(t, sweep["f1"], label="F1")
        ax.axvline(breakeven, color="grey", linestyle=":",
                   label=f"Break-even p = {breakeven:.2f}")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Threshold")
        ax.set_title(label, fontsize=9)
        ax2 = ax.twinx()
        ax2.plot(t, sweep["net_benefit_per_1000"], color="#C44E52", linewidth=1.2,
                 label="Net benefit per 1,000 (placeholder costs)")
        ax2.set_ylabel(f"Net benefit per 1,000 customers ({business.currency})", fontsize=8)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="center right")
    fig.suptitle("Threshold analysis on the validation split", y=1.0)
    return _save(fig, path)


def plot_learning_curves(curves: dict, path: Path | str) -> Path:
    """Training vs cross-validated PR-AUC as training size grows."""
    fig, flat = _grid(len(curves))
    for ax, (label, c) in zip(flat, curves.items()):
        sizes = c["sizes"]
        ax.plot(sizes, c["train_mean"], marker="o", label="Training")
        ax.fill_between(sizes, c["train_mean"] - c["train_std"],
                        c["train_mean"] + c["train_std"], alpha=0.2)
        ax.plot(sizes, c["cv_mean"], marker="o", label="Cross-validation")
        ax.fill_between(sizes, c["cv_mean"] - c["cv_std"],
                        c["cv_mean"] + c["cv_std"], alpha=0.2)
        ax.set_title(_label(label))
        ax.set_xlabel("Training rows")
        ax.set_ylabel("PR-AUC")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Learning curves (training split, tuned models)", y=1.0)
    return _save(fig, path)