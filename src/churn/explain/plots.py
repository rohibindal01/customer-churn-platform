"""Figures for explainability."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_bar(labels, values, path: Path | str, title: str, xlabel: str,
             errors=None, color: str = "#4C72B0") -> Path:
    """Horizontal bars, first item at the top."""
    labels = list(labels)[::-1]
    values = np.asarray(values, dtype=float)[::-1]
    errors = None if errors is None else np.asarray(errors, dtype=float)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.35 * len(labels) + 1.6))
    ax.barh(labels, values, xerr=errors, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    return _save(fig, path)


def plot_signed_bar(labels, values, path: Path | str, title: str, xlabel: str) -> Path:
    """Bars coloured by sign: red raises churn risk, blue lowers it."""
    labels = list(labels)[::-1]
    values = np.asarray(values, dtype=float)[::-1]
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in values]
    fig, ax = plt.subplots(figsize=(8.5, 0.35 * len(labels) + 1.6))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    return _save(fig, path)


def plot_shap_summary(shap_values: np.ndarray, X_t: np.ndarray, feature_names, path: Path | str,
                      units: str = "log-odds", top_n: int = 15, seed: int = 0) -> Path:
    """One row per feature, one dot per customer; colour = feature value (low to high)."""
    order = np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:top_n]
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(order) + 1.8))
    scatter = None
    for row, j in enumerate(order[::-1]):
        values = X_t[:, j]
        lo, hi = np.percentile(values, [5, 95])
        colors = (np.clip((values - lo) / (hi - lo), 0, 1) if hi > lo
                  else np.full(len(values), 0.5))
        jitter = rng.uniform(-0.3, 0.3, len(values))
        scatter = ax.scatter(shap_values[:, j], row + jitter, c=colors, cmap="coolwarm",
                             vmin=0, vmax=1, s=6, alpha=0.6, linewidths=0)
    ax.set_yticks(range(len(order)), [feature_names[j] for j in order[::-1]])
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel(f"SHAP value ({units}); right = higher predicted churn risk")
    ax.set_title("SHAP summary (validation split)")
    if scatter is not None:
        bar = fig.colorbar(scatter, ax=ax)
        bar.set_ticks([0, 1])
        bar.set_ticklabels(["low", "high"])
        bar.set_label("Feature value")
    return _save(fig, path)


def plot_local_explanation(contributions, path: Path | str, title: str,
                           units: str = "log-odds") -> Path:
    """One customer's contributions: red pushes towards churn, blue away from it."""
    labels = [f"{r.group} = {r.value}" if r.value != "" else r.group
              for r in contributions.itertuples(index=False)]
    return plot_signed_bar(labels, contributions["shap"], path, title,
                           f"Contribution to the prediction ({units})")