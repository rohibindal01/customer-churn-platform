"""Tests for the metrics helpers, using hand-computed values."""
import math

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from churn.models.evaluation import classification_report_text, compute_metrics, get_scores


def test_known_confusion_matrix_values():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]
    y_score = [0.9, 0.4, 0.2, 0.1]
    m = compute_metrics(y_true, y_pred, y_score)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 0, 1, 1)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx(0.75)
    assert m["mcc"] == pytest.approx(2 / math.sqrt(12))
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["brier"] == pytest.approx(0.105)


def test_perfect_predictions():
    m = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    for name in ("accuracy", "precision", "recall", "f1", "specificity", "mcc", "roc_auc"):
        assert m[name] == pytest.approx(1.0)


def test_brier_is_skipped_for_decision_scores():
    m = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1], [-2.0, 1.5, -1.0, 2.0],
                        score_is_probability=False)
    assert math.isnan(m["brier"])
    assert m["roc_auc"] == pytest.approx(1.0)


def test_get_scores_uses_probability_or_decision_function():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] > 0).astype(int)
    scores, is_probability = get_scores(LogisticRegression().fit(X, y), X)
    assert is_probability and 0 <= scores.min() and scores.max() <= 1
    scores, is_probability = get_scores(SVC(probability=False).fit(X, y), X)
    assert not is_probability and scores.min() < 0 < scores.max()


def test_classification_report_names_both_classes():
    text = classification_report_text([0, 1, 0, 1], [0, 1, 1, 1])
    assert "Churned" in text and "Retained" in text