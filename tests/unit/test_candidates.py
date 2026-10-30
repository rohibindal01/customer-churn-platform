"""Tests for candidate evaluation, using the synthetic 400-row dataset."""
import joblib
import numpy as np
import pytest

from churn.config.settings import BusinessSettings
from churn.features.splitting import split_xy, stratified_split
from churn.models.calibration import CALIBRATION_METHODS
from churn.models.calibration_plots import (
    plot_learning_curves,
    plot_reliability,
    plot_threshold_curves,
)
from churn.models.candidates import (
    best_calibration_per_model,
    comparison_table,
    compute_learning_curve,
    evaluate_candidates,
    format_candidates_report,
    load_tuned_pipelines,
    save_candidates,
    threshold_tables,
)
from churn.models.pipeline import build_model_pipeline

BUSINESS = BusinessSettings(customer_value=500.0, offer_cost=50.0, save_rate=0.30)


@pytest.fixture
def data(synthetic_df):
    train, val, _ = stratified_split(synthetic_df)
    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    pipe = build_model_pipeline("logistic_regression", "none", seed=42).fit(X_train, y_train)
    return {"logistic_regression": pipe}, X_train, y_train, X_val, y_val


@pytest.fixture
def results(data):
    tuned, X_train, y_train, X_val, y_val = data
    return evaluate_candidates(tuned, X_train, y_train, X_val, y_val, n_splits=3, seed=42)


def test_every_calibration_method_is_evaluated(results, data):
    X_val = data[3]
    assert [r.calibration for r in results] == list(CALIBRATION_METHODS)
    assert all(r.status == "ok" for r in results)
    assert all(r.val_prob.shape == (len(X_val),) for r in results)


def test_failure_is_recorded(data):
    _, X_train, y_train, X_val, y_val = data
    out = evaluate_candidates({"broken": "not a model"}, X_train, y_train, X_val, y_val,
                              methods=["uncalibrated"], n_splits=3)
    assert out[0].status == "failed" and out[0].reason


def test_comparison_and_best_calibration(results):
    table = comparison_table(results)
    assert len(table) == 3
    assert {"roc_auc", "pr_auc", "brier", "log_loss", "ece"} <= set(table.columns)
    best = best_calibration_per_model(table)
    assert len(best) == 1
    assert best.iloc[0]["brier"] == pytest.approx(table["brier"].min())


def test_threshold_tables(results, data):
    sweep, choices = threshold_tables(results, data[4], BUSINESS, recall_target=0.8)
    assert len(sweep) == 3 * 99
    assert len(choices) == 3 * 4
    assert {"model", "calibration", "policy", "threshold"} <= set(choices.columns)


def test_save_and_reload(results, data, tmp_path):
    X_val, y_val = data[3], data[4]
    table = comparison_table(results)
    sweep, choices = threshold_tables(results, y_val, BUSINESS, recall_target=0.8)
    written = save_candidates(results, table, sweep, choices, y_val,
                              tmp_path / "metrics", tmp_path / "models", meta={"x": 1})
    assert all(path.exists() and path.stat().st_size > 0 for path in written)
    loaded = joblib.load(tmp_path / "models" / "logistic_regression__sigmoid.joblib")
    np.testing.assert_allclose(loaded.predict_proba(X_val)[:, 1], results[1].val_prob)
    probs = np.load(tmp_path / "metrics" / "validation_probabilities.npz")
    assert "y_val" in probs.files and "logistic_regression__isotonic" in probs.files


def test_plots_and_report(results, data, tmp_path):
    tuned, X_train, y_train, X_val, y_val = data
    table = comparison_table(results)
    sweep, choices = threshold_tables(results, y_val, BUSINESS, recall_target=0.8)
    one = sweep[(sweep["model"] == "logistic_regression") & (sweep["calibration"] == "sigmoid")]
    curve = compute_learning_curve(tuned["logistic_regression"], X_train, y_train, n_splits=3)
    assert len(curve["sizes"]) == 5
    paths = [
        plot_reliability(results, y_val, tmp_path / "reliability.png"),
        plot_threshold_curves({"LR (sigmoid)": one}, BUSINESS, tmp_path / "thresholds.png"),
        plot_learning_curves({"logistic_regression": curve}, tmp_path / "learning.png"),
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    report = format_candidates_report(table, choices, results, BUSINESS, recall_target=0.8)
    assert "PLACEHOLDER" in report and "logistic_regression" in report


def test_load_tuned_pipelines(tmp_path, data):
    with pytest.raises(FileNotFoundError):
        load_tuned_pipelines(tmp_path / "missing")
    joblib.dump(data[0]["logistic_regression"], tmp_path / "logistic_regression.joblib")
    assert list(load_tuned_pipelines(tmp_path)) == ["logistic_regression"]