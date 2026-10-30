"""Every model pipeline survives a joblib save/load round trip unchanged."""
import joblib
import numpy as np
import pytest

from churn.features.splitting import split_xy
from churn.models.calibration import build_calibrated
from churn.models.evaluation import get_scores
from churn.models.pipeline import build_model_pipeline
from churn.models.registry import list_model_keys


@pytest.mark.parametrize("model_key", list_model_keys())
def test_model_pipeline_round_trip(synthetic_df, model_key, tmp_path):
    X, y = split_xy(synthetic_df)
    pipe = build_model_pipeline(model_key, "none", seed=42).fit(X, y)
    path = tmp_path / f"{model_key}.joblib"
    joblib.dump(pipe, path)
    loaded = joblib.load(path)

    before, kind_before = get_scores(pipe, X)
    after, kind_after = get_scores(loaded, X)
    assert kind_before == kind_after
    np.testing.assert_allclose(before, after)
    np.testing.assert_array_equal(pipe.predict(X), loaded.predict(X))


def test_calibrated_pipeline_round_trip(synthetic_df, tmp_path):
    X, y = split_xy(synthetic_df)
    model = build_calibrated(build_model_pipeline("logistic_regression", "none", seed=42),
                             "sigmoid", n_splits=3).fit(X, y)
    path = tmp_path / "calibrated.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(model.predict_proba(X), joblib.load(path).predict_proba(X))