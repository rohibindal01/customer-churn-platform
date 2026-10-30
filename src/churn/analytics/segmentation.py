"""Customer segmentation with K-Means, compared with hierarchical clustering and DBSCAN.

Segments use customer attributes only. The churn label is NOT an input; it
is used afterwards to profile each segment, which keeps segmentation
separate from the supervised churn model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from churn.features.cleaning import clean_features
from churn.features.engineering import engineer_features

CONTRACT_LEVEL = {"Month-to-month": 0.0, "One year": 1.0, "Two year": 2.0}
SEGMENT_FEATURES = ["tenure", "MonthlyCharges", "log_TotalCharges", "addon_count",
                    "contract_level", "internet_dsl", "internet_fiber", "internet_none"]
K_RANGE = tuple(range(2, 11))
BUSINESS_K = (3, 8)            # fewer is too coarse to act on, more is too many to manage
MIN_STABILITY = 0.80
MIN_CLUSTER_PCT = 5.0
WEAK_SILHOUETTE = 0.25
STRONG_SILHOUETTE = 0.50


def _flag(condition: pd.Series) -> np.ndarray:
    return condition.fillna(False).to_numpy(dtype=float)


def build_segment_features(X_raw: pd.DataFrame) -> pd.DataFrame:
    """Readable segmentation inputs from raw customer rows (no churn label)."""
    eng = engineer_features(clean_features(X_raw))
    total = eng["TotalCharges"].fillna(eng["tenure"] * eng["MonthlyCharges"]).clip(lower=0)
    out = pd.DataFrame({
        "tenure": eng["tenure"].to_numpy(dtype=float),
        "MonthlyCharges": eng["MonthlyCharges"].to_numpy(dtype=float),
        "log_TotalCharges": np.log1p(total.to_numpy(dtype=float)),
        "addon_count": eng["addon_count"].to_numpy(dtype=float),
        "contract_level": eng["Contract"].map(CONTRACT_LEVEL).to_numpy(dtype=float),
        "internet_dsl": _flag(eng["InternetService"] == "DSL"),
        "internet_fiber": _flag(eng["InternetService"] == "Fiber optic"),
        "internet_none": _flag(eng["InternetService"] == "No"),
    }, index=X_raw.index)
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(f"Segmentation inputs contain missing or unknown values in {bad}")
    return out


def structure_label(silhouette: float) -> str:
    if silhouette >= STRONG_SILHOUETTE:
        return "strong"
    if silhouette >= WEAK_SILHOUETTE:
        return "reasonable"
    return "weak"


def _silhouette(X: np.ndarray, labels: np.ndarray, seed: int) -> float:
    sample = None if len(X) <= 5000 else 5000
    return float(silhouette_score(X, labels, sample_size=sample, random_state=seed))


def evaluate_k(X_scaled, ks=K_RANGE, seed: int = 42, n_stability: int = 10) -> pd.DataFrame:
    """Elbow, silhouette, other indices and bootstrap stability for every k."""
    X = np.asarray(X_scaled, dtype=float)
    n = len(X)
    rng = np.random.default_rng(seed)
    resamples = [rng.choice(n, size=n, replace=True) for _ in range(n_stability)]
    rows = []
    for k in ks:
        reference = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        labels = reference.labels_
        aris = [adjusted_rand_score(labels, KMeans(n_clusters=k, n_init=3,
                                                   random_state=seed + i + 1)
                                    .fit(X[idx]).predict(X))
                for i, idx in enumerate(resamples)]
        sizes = np.bincount(labels, minlength=k)
        rows.append({
            "k": int(k),
            "inertia": float(reference.inertia_),
            "silhouette": _silhouette(X, labels, seed),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "stability_ari": float(np.mean(aris)) if aris else float("nan"),
            "smallest_cluster_pct": 100.0 * sizes.min() / n,
        })
    return pd.DataFrame(rows)


def choose_k(table: pd.DataFrame, k_range=BUSINESS_K, min_stability: float = MIN_STABILITY,
             min_cluster_pct: float = MIN_CLUSTER_PCT) -> dict:
    """Highest silhouette among stable, non-tiny solutions within the business range."""
    low, high = k_range
    candidates = table[(table["k"] >= low) & (table["k"] <= high)]
    passing = candidates[(candidates["stability_ari"] >= min_stability)
                         & (candidates["smallest_cluster_pct"] >= min_cluster_pct)]
    pool = passing if not passing.empty else candidates
    best = pool.loc[pool["silhouette"].idxmax()]
    silhouette = float(best["silhouette"])
    reason = (f"highest silhouette ({silhouette:.3f}) for k in {low}..{high} with stability >= "
              f"{min_stability} and every cluster >= {min_cluster_pct:.0f}% of customers")
    if passing.empty:
        reason = (f"no k in {low}..{high} met the stability and size constraints; "
                  f"highest silhouette ({silhouette:.3f}) shown, treat the segments with caution")
    return {"k": int(best["k"]), "silhouette": silhouette,
            "structure": structure_label(silhouette), "reason": reason,
            "constraints_met": not passing.empty}


@dataclass
class SegmentModel:
    """Scaler + K-Means; segments are numbered from shortest to longest tenure."""
    scaler: StandardScaler
    kmeans: KMeans
    label_map: dict
    spend_cuts: tuple
    names: dict = field(default_factory=dict)
    feature_columns: tuple = tuple(SEGMENT_FEATURES)

    @property
    def k(self) -> int:
        return int(self.kmeans.n_clusters)

    def transform(self, X_raw: pd.DataFrame) -> np.ndarray:
        features = build_segment_features(X_raw)[list(self.feature_columns)]
        return self.scaler.transform(features)

    def predict(self, X_raw: pd.DataFrame) -> np.ndarray:
        raw = self.kmeans.predict(self.transform(X_raw))
        return np.array([self.label_map[int(r)] for r in raw], dtype=int)

    def segment_names(self, X_raw: pd.DataFrame) -> list[str]:
        return [self.names.get(int(s), f"S{int(s) + 1}") for s in self.predict(X_raw)]


def fit_segment_model(X_raw: pd.DataFrame, k: int, seed: int = 42) -> SegmentModel:
    """Fit the scaler and K-Means on raw training rows."""
    features = build_segment_features(X_raw)[SEGMENT_FEATURES]
    scaler = StandardScaler().fit(features)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaler.transform(features))
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    order = np.argsort(centers[:, SEGMENT_FEATURES.index("tenure")], kind="stable")
    label_map = {int(old): new for new, old in enumerate(order)}
    cuts = tuple(float(q) for q in np.quantile(features["MonthlyCharges"], [1 / 3, 2 / 3]))
    return SegmentModel(scaler=scaler, kmeans=kmeans, label_map=label_map, spend_cuts=cuts)


def _method_row(name: str, labels: np.ndarray, X: np.ndarray, reference: np.ndarray) -> dict:
    clusters = sorted(set(labels.tolist()) - {-1})
    clustered = labels != -1
    silhouette = (float(silhouette_score(X[clustered], labels[clustered]))
                  if len(clusters) >= 2 and clustered.sum() > len(clusters) else float("nan"))
    return {"method": name, "clusters": len(clusters),
            "noise_pct": 100.0 * float((labels == -1).mean()),
            "silhouette": silhouette,
            "ari_vs_kmeans": float(adjusted_rand_score(reference, labels))}


def compare_methods(X_scaled, kmeans_labels, k: int, seed: int = 42) -> pd.DataFrame:
    """K-Means vs hierarchical (Ward) vs DBSCAN on the same scaled data."""
    X = np.asarray(X_scaled, dtype=float)
    reference = np.asarray(kmeans_labels)
    rows = [_method_row(f"K-Means (k={k})", reference, X, reference)]
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    rows.append(_method_row(f"Hierarchical, Ward (k={k})", ward, X, reference))
    min_samples = 2 * X.shape[1]
    distances = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)[0][:, -1]
    eps = float(np.quantile(distances, 0.90))
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    rows.append(_method_row(f"DBSCAN (eps={eps:.2f}, min_samples={min_samples})",
                            dbscan, X, reference))
    return pd.DataFrame(rows)


_PCT_COLUMNS = {
    "month_to_month_pct": "month_to_month", "two_year_pct": "two_year",
    "fiber_pct": "fiber", "dsl_pct": "dsl", "no_internet_pct": "no_internet",
    "no_security_or_support_pct": "no_security_or_support",
    "electronic_check_pct": "electronic_check", "senior_pct": "senior",
}


def profile_segments(X_raw: pd.DataFrame, labels, churn=None, probability=None) -> pd.DataFrame:
    """Size, averages, shares and (optionally) churn and predicted risk per segment."""
    eng = engineer_features(clean_features(X_raw)).reset_index(drop=True)
    frame = pd.DataFrame({
        "segment": np.asarray(labels).astype(int),
        "tenure": eng["tenure"].to_numpy(dtype=float),
        "MonthlyCharges": eng["MonthlyCharges"].to_numpy(dtype=float),
        "TotalCharges": eng["TotalCharges"].to_numpy(dtype=float),
        "addon_count": eng["addon_count"].to_numpy(dtype=float),
        "month_to_month": _flag(eng["Contract"] == "Month-to-month"),
        "two_year": _flag(eng["Contract"] == "Two year"),
        "fiber": _flag(eng["InternetService"] == "Fiber optic"),
        "dsl": _flag(eng["InternetService"] == "DSL"),
        "no_internet": _flag(eng["InternetService"] == "No"),
        "no_security_or_support": eng["no_security_or_support"].to_numpy(dtype=float),
        "electronic_check": _flag(eng["PaymentMethod"] == "Electronic check"),
        "senior": _flag(eng["SeniorCitizen"] == 1),
    })
    spec = {"customers": ("tenure", "size"), "tenure_mean": ("tenure", "mean"),
            "monthly_mean": ("MonthlyCharges", "mean"), "total_mean": ("TotalCharges", "mean"),
            "addons_mean": ("addon_count", "mean")}
    spec.update({name: (column, "mean") for name, column in _PCT_COLUMNS.items()})
    if churn is not None:
        frame["churn"] = np.asarray(churn).astype(float)
        spec["churn_rate_pct"] = ("churn", "mean")
    if probability is not None:
        frame["predicted"] = np.asarray(probability, dtype=float)
        spec["mean_predicted_pct"] = ("predicted", "mean")

    profile = frame.groupby("segment").agg(**spec).reset_index()
    for column in [*_PCT_COLUMNS, "churn_rate_pct", "mean_predicted_pct"]:
        if column in profile.columns:
            profile[column] = profile[column] * 100.0
    profile.insert(2, "share_pct", 100.0 * profile["customers"] / profile["customers"].sum())
    if churn is not None:
        overall = frame["churn"].mean() * 100.0
        profile["lift"] = profile["churn_rate_pct"] / overall if overall > 0 else float("nan")
    return profile


def name_segments(profile: pd.DataFrame, spend_cuts) -> dict[int, str]:
    """Readable names generated from each segment's averages and shares."""
    low_cut, high_cut = spend_cuts
    names = {}
    for row in profile.itertuples(index=False):
        tenure = ("New" if row.tenure_mean < 12
                  else "Established" if row.tenure_mean <= 48 else "Long-standing")
        spend = ("low-spend" if row.monthly_mean < low_cut
                 else "mid-spend" if row.monthly_mean < high_cut else "high-spend")
        if row.no_internet_pct >= 50:
            internet = "no internet"
        elif row.fiber_pct >= 50:
            internet = "fiber"
        elif row.dsl_pct >= 50:
            internet = "DSL"
        else:
            internet = "mixed internet"
        if row.month_to_month_pct >= 50:
            contract = "month-to-month"
        elif row.two_year_pct >= 50:
            contract = "two-year"
        else:
            contract = "mixed contracts"
        segment = int(row.segment)
        names[segment] = f"S{segment + 1}: {tenure}, {spend}, {internet}, {contract}"
    return names