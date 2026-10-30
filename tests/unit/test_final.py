"""Tests for the final evaluation and saving helpers."""
import json

import joblib
import numpy as np
import pytest

from churn.config.settings import BusinessSettings
from churn.features.preprocessing import get_feature_names
from churn.features.splitting import split_xy, stratified_split
from churn.models.calibration import build_calibrated
from churn.models.final import (
    FINAL_MODEL_FILE,
    METADATA_FILE,
    base_pipeline,
    check_not_finalized,
    evaluate_on_test,
    metrics_at_threshold,
    plot_test_calibration,
    plot_test_confusion,
    plot_test_curves,
    save_final,
)
from churn.models.pipeline import build_model_pipeline

BUSINESS = BusinessSettings(customer_value=500.0, offer_cost=50.0, save_rate=0.30)


@pytest.fixture
def fitted(synthetic_df):
    train, _, test = stratified_split(synthetic_df)
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)
    plain = build_model_pipeline("logistic_regression", "none", seed=42).fit(X_train, y_train)
    calibrated = build_calibrated(build_model_pipeline("logistic_regression", "none", seed=42),
                                  "sigmoid", n_splits=3).fit(X_train, y_train)
    return plain, calibrated, X_test, y_test


def test_base_pipeline(fitted):
    plain, calibrated, _, _ = fitted
    assert base_pipeline(plain) is plain
    inner = base_pipeline(calibrated)
    assert "preprocess" in inner.named_steps
    assert len(get_feature_names(inner)) > 0


def test_metrics_at_threshold_hand_computed():
    m = metrics_at_threshold([1, 1, 0, 0], [0.9, 0.6, 0.4, 0.1], 0.5, BUSINESS)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 0, 0, 2)
    assert m["flagged_pct"] == pytest.approx(50.0)
    assert m["net_benefit"] == pytest.approx(200.0)


def test_evaluate_on_test(fitted):
    plain, _, X_test, y_test = fitted
    thresholds = {"default_0.5": 0.5, "max_f1": 0.3}
    ev = evaluate_on_test(plain, X_test, y_test, thresholds, "max_f1", BUSINESS, n_boot=100)
    assert set(ev["policies"]) == set(thresholds)
    assert ev["chosen_threshold"] == pytest.approx(0.3)
    ci = ev["confidence_intervals"]["roc_auc"]
    assert ci["ci_low"] <= ci["value"] <= ci["ci_high"]
    assert ev["probabilities"].shape == (len(X_test),)
    assert "Churned" in ev["classification_report"]


def test_save_final_and_reload(fitted, tmp_path):
    plain, _, X_test, y_test = fitted
    ev = evaluate_on_test(plain, X_test, y_test, {"max_f1": 0.3}, "max_f1", BUSINESS, n_boot=50)
    written = save_final(plain, {"model": "logistic_regression", "value": np.float64(0.5)},
                         ev, tmp_path / "final", tmp_path / "metrics")
    assert all(path.exists() and path.stat().st_size > 0 for path in written)
    loaded = joblib.load(tmp_path / "final" / FINAL_MODEL_FILE)
    np.testing.assert_allclose(loaded.predict_proba(X_test), plain.predict_proba(X_test))
    meta = json.loads((tmp_path / "final" / METADATA_FILE).read_text(encoding="utf-8"))
    assert meta["value"] == pytest.approx(0.5)
    summary = json.loads((tmp_path / "metrics" / "test_metrics.json").read_text(encoding="utf-8"))
    assert "probabilities" not in summary and "policies" in summary


def test_check_not_finalized(tmp_path):
    assert check_not_finalized(tmp_path, force=False) == 0
    (tmp_path / METADATA_FILE).write_text(json.dumps({"test_evaluation_count": 1}),
                                          encoding="utf-8")
    with pytest.raises(FileExistsError):
        check_not_finalized(tmp_path, force=False)
    assert check_not_finalized(tmp_path, force=True) == 1


def test_final_plots(fitted, tmp_path):
    plain, _, X_test, y_test = fitted
    prob = plain.predict_proba(X_test)[:, 1]
    y = np.asarray(y_test)
    paths = [
        plot_test_curves(y, prob, 0.3, tmp_path / "curves.png"),
        plot_test_calibration(y, prob, tmp_path / "calibration.png"),
        plot_test_confusion(y, prob, 0.3, tmp_path / "confusion.png"),
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)