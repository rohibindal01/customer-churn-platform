"""Tests for the pre-registered selection rule."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from churn.models.selection import (
    BRIER_TOLERANCE,
    COMPLEXITY_ORDER,
    bootstrap_ci,
    candidate_variants,
    paired_bootstrap_diff,
    select_final,
    selection_table,
)
from churn.models.tuning import SHORTLIST


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    n = 800
    y = (rng.random(n) < 0.3).astype(int)
    good = np.clip(0.45 * y + 0.55 * rng.random(n), 0, 1)
    same_ranking = 0.9 * good + 0.05          # monotonic: identical ranking to `good`
    bad = rng.random(n)
    return y, good, same_ranking, bad


def _variants(y, good, same_ranking, bad, lr_brier=0.138):
    return pd.DataFrame({
        "model": ["logistic_regression", "gradient_boosting", "catboost"],
        "calibration": ["uncalibrated"] * 3,
        "roc_auc": [0.83, 0.84, 0.60],
        "pr_auc": [0.650, 0.655, average_precision_score(y, bad)],   # GB is the reference
        "brier": [lr_brier, 0.136, 0.137],
        "ece": [0.03, 0.02, 0.03],
        "variant_reason": ["test"] * 3,
    })


def _probs(good, same_ranking, bad):
    return {"logistic_regression__uncalibrated": same_ranking,
            "gradient_boosting__uncalibrated": good,
            "catboost__uncalibrated": bad}


def test_complexity_order_covers_the_shortlist():
    assert set(COMPLEXITY_ORDER) == set(SHORTLIST)


def test_variant_rule():
    comparison = pd.DataFrame({
        "model": ["adaboost", "adaboost", "adaboost", "gradient_boosting",
                  "gradient_boosting", "logistic_regression"],
        "calibration": ["uncalibrated", "sigmoid", "isotonic", "uncalibrated",
                        "sigmoid", "uncalibrated"],
        "roc_auc": [0.84] * 6,
        "pr_auc": [0.65, 0.65, 0.64, 0.66, 0.66, 0.655],
        "brier": [0.170, 0.138, 0.137, 0.1364, 0.13635, 0.138],
        "ece": [0.17, 0.02, 0.03, 0.02, 0.017, 0.03],
    })
    v = candidate_variants(comparison).set_index("model")
    assert v.loc["adaboost", "calibration"] == "sigmoid"
    assert v.loc["gradient_boosting", "calibration"] == "uncalibrated"
    assert v.loc["logistic_regression", "calibration"] == "uncalibrated"
    assert "isotonic" not in set(v["calibration"])


def test_bootstrap_ci_contains_point_estimate(scores):
    y, good, _, _ = scores
    ci = bootstrap_ci(y, good, "pr_auc", n_boot=300, seed=1)
    assert ci["ci_low"] <= ci["value"] <= ci["ci_high"]


def test_paired_diff_of_identical_rankings_is_zero(scores):
    y, good, same_ranking, _ = scores
    d = paired_bootstrap_diff(y, good, same_ranking, n_boot=200)
    assert d["diff"] == pytest.approx(0.0)
    assert d["ci_low"] == pytest.approx(0.0) and d["ci_high"] == pytest.approx(0.0)


def test_paired_diff_detects_a_clearly_worse_model(scores):
    y, good, _, bad = scores
    assert paired_bootstrap_diff(y, good, bad, n_boot=300)["ci_low"] > 0


def test_rule_picks_the_simplest_tied_variant(scores):
    y, good, same_ranking, bad = scores
    table = selection_table(_variants(y, good, same_ranking, bad),
                            _probs(good, same_ranking, bad), y, n_boot=300, seed=0)
    by = table.set_index("model")
    assert bool(by.loc["gradient_boosting", "is_reference"])
    assert not bool(by.loc["catboost", "eligible"])
    decision = select_final(table)
    assert decision["model"] == "logistic_regression"
    assert decision["selected_by"] == "rule"
    assert "statistically tied" in decision["reason"]


def test_brier_tolerance_excludes_poor_probabilities(scores):
    y, good, same_ranking, bad = scores
    variants = _variants(y, good, same_ranking, bad, lr_brier=0.136 + BRIER_TOLERANCE + 0.001)
    table = selection_table(variants, _probs(good, same_ranking, bad), y, n_boot=300, seed=0)
    assert select_final(table)["model"] == "gradient_boosting"


def test_manual_override():
    decision = select_final(pd.DataFrame(), model="catboost", calibration="sigmoid")
    assert decision["model"] == "catboost" and decision["calibration"] == "sigmoid"
    assert decision["selected_by"] == "manual override"


def test_missing_probabilities_raise(scores):
    y, good, same_ranking, bad = scores
    with pytest.raises(KeyError):
        selection_table(_variants(y, good, same_ranking, bad), {}, y, n_boot=10)