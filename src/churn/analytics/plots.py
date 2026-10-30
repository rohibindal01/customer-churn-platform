"""Figures for segmentation, risk bands and recommendations."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from churn.analytics.segmentation import MIN_STABILITY, WEAK_SILHOUETTE  # noqa: E402


def _save(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_k_selection(table, chosen_k: int, path: Path | str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(table["k"], table["inertia"], marker="o")
    axes[0].set_title("Elbow: within-cluster sum of squares")
    axes[1].plot(table["k"], table["silhouette"], marker="o")
    axes[1].axhline(WEAK_SILHOUETTE, color="grey", linestyle=":", label="weak below 0.25")
    axes[1].set_title("Silhouette (higher is better)")
    axes[1].legend(fontsize=8)
    axes[2].plot(table["k"], table["stability_ari"], marker="o")
    axes[2].axhline(MIN_STABILITY, color="grey", linestyle=":", label=f"required {MIN_STABILITY}")
    axes[2].set_title("Bootstrap stability (ARI)")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.axvline(chosen_k, color="#C44E52", linestyle="--")
        ax.set_xlabel("k (number of segments)")
    return _save(fig, path)


def plot_segments_pca(X_scaled, labels, names: dict, path: Path | str, seed: int = 42,
                      max_points: int = 3000) -> Path:
    X = np.asarray(X_scaled, dtype=float)
    labels = np.asarray(labels)
    pca = PCA(n_components=2, random_state=seed).fit(X)
    coords = pca.transform(X)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(max_points, len(X)), replace=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    for segment in sorted(set(labels.tolist())):
        mask = labels[idx] == segment
        ax.scatter(coords[idx][mask, 0], coords[idx][mask, 1], s=8, alpha=0.5,
                   label=names.get(segment, f"S{segment + 1}"))
    ratio = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ratio[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({ratio[1]:.0%} of variance)")
    ax.set_title("Segments projected to 2 dimensions (PCA, training split)")
    ax.legend(fontsize=7, loc="best", markerscale=2)
    return _save(fig, path)


def plot_segment_churn(profile, overall_pct: float, path: Path | str) -> Path:
    data = profile.sort_values("churn_rate_pct")
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(data) + 1.8))
    bars = ax.barh(data["name"], data["churn_rate_pct"], color="#4C72B0")
    for bar, n in zip(bars, data["customers"]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  n={int(n)}",
                va="center", fontsize=8)
    ax.axvline(overall_pct, color="#C44E52", linestyle="--", label=f"Overall {overall_pct:.1f}%")
    ax.set_xlabel("Churn rate (%)")
    ax.set_title("Churn rate by segment (training split)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_segment_heatmap(profile, path: Path | str) -> Path:
    columns = ["tenure_mean", "monthly_mean", "addons_mean", "month_to_month_pct", "fiber_pct",
               "no_internet_pct", "no_security_or_support_pct", "electronic_check_pct",
               "senior_pct", "churn_rate_pct"]
    columns = [c for c in columns if c in profile.columns]
    data = profile[columns]
    spread = data.std(ddof=0).replace(0, 1)
    z = ((data - data.mean()) / spread).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.2 * len(columns) + 3, 0.6 * len(profile) + 2))
    ax.imshow(z, cmap="coolwarm", aspect="auto", vmin=-2, vmax=2)
    for (i, j), value in np.ndenumerate(data.to_numpy(dtype=float)):
        ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(columns)), columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(profile)), profile["name"], fontsize=8)
    ax.set_title("Segment profiles (colour = relative to other segments; numbers = actual)")
    return _save(fig, path)


def plot_risk_bands(summary, path: Path | str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(summary["band"], summary["customers"], color="#4C72B0")
    ax.set_ylabel("Customers")
    ax2 = ax.twinx()
    ax2.plot(summary["band"], summary["mean_predicted_pct"], marker="o", color="#DD8452",
             label="Mean predicted churn %")
    if "observed_churn_pct" in summary.columns:
        ax2.plot(summary["band"], summary["observed_churn_pct"], marker="s", color="#C44E52",
                 label="Observed churn %")
    ax2.set_ylabel("Churn (%)")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left", fontsize=8)
    ax.set_title("Risk bands on the validation split")
    return _save(fig, path)


def plot_rule_frequency(counts, path: Path | str) -> Path:
    data = counts.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(data) + 1.6))
    ax.barh(data.index, data.values, color="#55A868")
    ax.set_xlabel("Customers for whom the rule fired")
    ax.set_title("Recommendation rules triggered (validation split)")
    return _save(fig, path)