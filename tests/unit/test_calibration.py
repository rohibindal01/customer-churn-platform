"""Tests for the calibration helpers."""
import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV

from churn.features.splitting import split_xy
from churn.models.calibration import (
    CALIBRATION_METHODS,
    build_calibrated,
    expected_calibration_error,
    probability_metrics,
)
from churn.models.pipeline import build_model_pipeline


def test_calibration_methods():
    assert CALIBRATION_METHODS == ("uncalibrated", "sigmoid", "isotonic")


def test_ece_is_zero_when_probabilities_match_reality():
    y = np.array([1, 0, 0, 0] * 25)
    assert expected_calibration_error(y, np.full(100, 0.25)) == pytest.approx(0.0)


def test_ece_measures_overconfidence():
    y = np.array([1, 0, 0, 0] * 25)
    assert expected_calibration_error(y, np.full(100, 0.9)) == pytest.approx(0.65)


def test_probability_metrics_hand_computed():
    m = probability_metrics([0, 1], [0.2, 0.6])
    assert m["brier"] == pytest.approx(0.1)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["mean_predicted"] == pytest.approx(0.4)
    assert m["observed_rate"] == pytest.approx(0.5)


def test_build_calibrated_types():
    pipe = build_model_pipeline("logistic_regression", "none", seed=42)
    plain = build_calibrated(pipe, "uncalibrated")
    assert plain is not pipe and type(plain) is type(pipe)
    wrapped = build_calibrated(pipe, "sigmoid")
    assert isinstance(wrapped, CalibratedClassifierCV)
    assert wrapped.method == "sigmoid" and wrapped.ensemble is False
    with pytest.raises(ValueError):
        build_calibrated(pipe, "magic")


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrated_pipeline_gives_valid_probabilities(synthetic_df, method):
    X, y = split_xy(synthetic_df)
    pipe = build_model_pipeline("logistic_regression", "none", seed=42)
    model = build_calibrated(pipe, method, n_splits=3).fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert p.shape == (len(X),)
    assert ((p >= 0) & (p <= 1)).all()