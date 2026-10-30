"""Contract tests on the deployed model, its metadata and its reported results."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from churn.config.settings import get_settings
from churn.data import schema
from churn.features.preprocessing import get_feature_names
from churn.models.final import FINAL_MODEL_FILE, METADATA_FILE, base_pipeline
from churn.models.registry import build_estimator

SETTINGS = get_settings()
FINAL = SETTINGS.models_dir / "final"
PORTFOLIO = SETTINGS.artifacts_dir / "serving" / "portfolio_validation.csv"
TEST_PROBS = SETTINGS.metrics_dir / "final" / "test_probabilities.npz"
TEST_METRICS = SETTINGS.metrics_dir / "final" / "test_metrics.json"

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (FINAL / FINAL_MODEL_FILE, FINAL / METADATA_FILE, PORTFOLIO)),
    reason="Final model or portfolio missing (run scripts 07 --evaluate-test and 11)",
)


@pytest.fixture(scope="module")
def model():
    return joblib.load(FINAL / FINAL_MODEL_FILE)


@pytest.fixture(scope="module")
def meta():
    return json.loads((FINAL / METADATA_FILE).read_text(encoding="utf-8"))


def test_reloaded_pipeline_reproduces_saved_predictions(model):
    portfolio = pd.read_csv(PORTFOLIO)
    probabilities = np.asarray(model.predict_proba(portfolio[schema.RAW_FEATURE_COLUMNS]))[:, 1]
    np.testing.assert_allclose(probabilities, portfolio["churn_probability"].to_numpy(dtype=float),
                               atol=1e-9)


def test_metadata_contract(model, meta):
    required = {"model", "calibration", "threshold", "input_columns", "feature_names",
                "validation_metrics", "test_metrics", "selection_rule", "data", "environment",
                "test_evaluation_count"}
    assert required <= set(meta)
    assert meta["input_columns"] == list(schema.RAW_FEATURE_COLUMNS)
    threshold = meta["threshold"]
    assert 0 < threshold["value"] < 1
    assert threshold["value"] == pytest.approx(threshold["all_policies"][threshold["policy"]])
    assert meta["feature_names"] == get_feature_names(base_pipeline(model))
    assert meta["test_evaluation_count"] >= 1
    for name, ci in meta["test_metrics"]["confidence_intervals"].items():
        assert ci["ci_low"] <= ci["value"] <= ci["ci_high"], name


def test_metadata_describes_the_deployed_model_type(model, meta):
    deployed = type(base_pipeline(model).steps[-1][1]).__name__
    expected = type(build_estimator(meta["model"], seed=42)).__name__
    assert deployed == expected


@pytest.mark.skipif(not (TEST_PROBS.exists() and TEST_METRICS.exists()),
                    reason="Saved test predictions missing")
def test_reported_test_metrics_match_saved_predictions():
    # Recomputed from the SAVED predictions: the test split is not scored again.
    saved = np.load(TEST_PROBS, allow_pickle=False)
    y, p = saved["y_true"], saved["probability"]
    reported = json.loads(TEST_METRICS.read_text(encoding="utf-8"))["confidence_intervals"]
    assert roc_auc_score(y, p) == pytest.approx(reported["roc_auc"]["value"], abs=1e-12)
    assert average_precision_score(y, p) == pytest.approx(reported["pr_auc"]["value"], abs=1e-12)