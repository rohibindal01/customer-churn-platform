"""Tests for customer segmentation."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from churn.analytics.plots import (
    plot_k_selection,
    plot_segment_churn,
    plot_segment_heatmap,
    plot_segments_pca,
)
from churn.analytics.segmentation import (
    SEGMENT_FEATURES,
    build_segment_features,
    choose_k,
    compare_methods,
    evaluate_k,
    fit_segment_model,
    name_segments,
    profile_segments,
)
from churn.features.splitting import split_xy


@pytest.fixture
def seg(synthetic_df):
    X, y = split_xy(synthetic_df)
    return X, y, fit_segment_model(X, 3, seed=0)


def test_segment_features(synthetic_df):
    X, _ = split_xy(synthetic_df)
    features = build_segment_features(X)
    assert list(features.columns) == SEGMENT_FEATURES
    assert len(features) == len(X)
    assert set(features["contract_level"].unique()) <= {0.0, 1.0, 2.0}
    internet = features[["internet_dsl", "internet_fiber", "internet_none"]].sum(axis=1)
    assert (internet == 1).all()


def test_unknown_contract_is_rejected(synthetic_df):
    X, _ = split_xy(synthetic_df)
    X = X.copy()
    X.loc[X.index[0], "Contract"] = "Weekly"
    with pytest.raises(ValueError):
        build_segment_features(X)


def test_evaluate_and_choose_k_on_clear_clusters():
    X, _ = make_blobs(n_samples=300, centers=3, n_features=8, cluster_std=0.5, random_state=0)
    table = evaluate_k(X, ks=range(2, 7), seed=0, n_stability=3)
    assert {"inertia", "silhouette", "stability_ari", "smallest_cluster_pct"} <= set(table.columns)
    choice = choose_k(table, k_range=(2, 6))
    assert choice["k"] == 3
    assert choice["structure"] == "strong" and choice["constraints_met"]


def test_choose_k_reports_unmet_constraints():
    table = pd.DataFrame({"k": [3, 4], "inertia": [10.0, 8.0], "silhouette": [0.20, 0.15],
                          "calinski_harabasz": [1.0, 1.0], "davies_bouldin": [1.0, 1.0],
                          "stability_ari": [0.5, 0.4], "smallest_cluster_pct": [2.0, 1.0]})
    choice = choose_k(table)
    assert choice["k"] == 3
    assert choice["constraints_met"] is False and choice["structure"] == "weak"


def test_segments_are_numbered_by_tenure(seg):
    _, _, model = seg
    centers = model.scaler.inverse_transform(model.kmeans.cluster_centers_)
    old_for_new = {new: old for old, new in model.label_map.items()}
    tenure = [centers[old_for_new[n], SEGMENT_FEATURES.index("tenure")] for n in range(model.k)]
    assert tenure == sorted(tenure)


def test_compare_methods(seg):
    X, _, model = seg
    table = compare_methods(model.transform(X), model.predict(X), model.k, seed=0)
    assert len(table) == 3
    assert table.iloc[0]["ari_vs_kmeans"] == pytest.approx(1.0)


def test_profile_and_names(seg):
    X, y, model = seg
    labels = model.predict(X)
    profile = profile_segments(X, labels, y, probability=np.full(len(X), 0.25))
    assert profile["customers"].sum() == len(X)
    assert profile["share_pct"].sum() == pytest.approx(100.0)
    assert profile["churn_rate_pct"].between(0, 100).all()
    assert profile["mean_predicted_pct"].tolist() == pytest.approx([25.0] * model.k)
    names = name_segments(profile, model.spend_cuts)
    assert len(set(names.values())) == model.k
    assert names[0].startswith("S1:")


def test_model_round_trip(seg, tmp_path):
    X, _, model = seg
    joblib.dump(model, tmp_path / "segments.joblib")
    loaded = joblib.load(tmp_path / "segments.joblib")
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))


def test_segmentation_plots(seg, tmp_path):
    X, y, model = seg
    scaled = model.transform(X)
    labels = model.predict(X)
    profile = profile_segments(X, labels, y)
    model.names = name_segments(profile, model.spend_cuts)
    profile.insert(1, "name", profile["segment"].map(model.names))
    table = evaluate_k(scaled, ks=range(2, 5), seed=0, n_stability=2)
    paths = [
        plot_k_selection(table, 3, tmp_path / "k.png"),
        plot_segments_pca(scaled, labels, model.names, tmp_path / "pca.png"),
        plot_segment_churn(profile, 25.0, tmp_path / "churn.png"),
        plot_segment_heatmap(profile, tmp_path / "heatmap.png"),
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)