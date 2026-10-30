"""Tests for the shared feature pipeline."""
import joblib
import numpy as np
import pytest

from churn.features.cleaning import clean_features
from churn.features.preprocessing import build_feature_pipeline, get_feature_names
from churn.features.splitting import split_xy, stratified_split


@pytest.fixture
def xy_splits(synthetic_df):
    train, val, _ = stratified_split(synthetic_df)
    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    return X_train, y_train, X_val, y_val


def test_output_has_no_nans_and_unique_names(xy_splits):
    X_train, _, X_val, _ = xy_splits
    pipe = build_feature_pipeline()
    train_out = pipe.fit_transform(X_train)
    val_out = pipe.transform(X_val)
    names = get_feature_names(pipe)
    assert train_out.shape[1] == val_out.shape[1] == len(names)
    assert len(set(names)) == len(names)
    assert not np.isnan(train_out).any()
    assert not np.isnan(val_out).any()


def test_scaler_is_fitted_on_training_data_only(xy_splits):
    X_train, _, _, _ = xy_splits
    pipe = build_feature_pipeline().fit(X_train)
    scaler = (
        pipe.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["scale"]
    )
    # tenure is the first numeric feature.
    assert scaler.mean_[0] == pytest.approx(clean_features(X_train)["tenure"].mean())


def test_unknown_category_at_inference_is_ignored(xy_splits):
    X_train, _, X_val, _ = xy_splits
    pipe = build_feature_pipeline().fit(X_train)
    X_bad = X_val.copy()
    X_bad["Contract"] = "Weekly"
    out = pipe.transform(X_bad)
    cols = [i for i, n in enumerate(get_feature_names(pipe)) if n.startswith("Contract_")]
    assert cols
    assert (out[:, cols] == 0).all()


def test_single_new_customer_with_blank_total_charges(xy_splits):
    X_train, _, X_val, _ = xy_splits
    pipe = build_feature_pipeline().fit(X_train)
    row = X_val.iloc[[0]].copy()
    row["tenure"] = 0
    row["TotalCharges"] = " "
    out = pipe.transform(row)
    assert out.shape == (1, len(get_feature_names(pipe)))
    assert not np.isnan(out).any()


def test_pipeline_survives_joblib_round_trip(xy_splits, tmp_path):
    X_train, _, X_val, _ = xy_splits
    pipe = build_feature_pipeline().fit(X_train)
    path = tmp_path / "feature_pipeline.joblib"
    joblib.dump(pipe, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(pipe.transform(X_val), loaded.transform(X_val))


def test_extra_columns_are_ignored(xy_splits):
    X_train, _, X_val, _ = xy_splits
    pipe = build_feature_pipeline().fit(X_train)
    X_extra = X_val.copy()
    X_extra["junk"] = 1
    np.testing.assert_allclose(pipe.transform(X_val), pipe.transform(X_extra))


def test_missing_feature_column_fails_loudly(xy_splits):
    X_train, _, X_val, _ = xy_splits
    pipe = build_feature_pipeline().fit(X_train)
    with pytest.raises(ValueError):
        pipe.transform(X_val.drop(columns=["Contract"]))