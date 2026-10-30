"""Metrics shared by the benchmark, tuning and the final evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_COLUMNS = [
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "specificity", "balanced_accuracy", "mcc", "brier",
]
CONFUSION_COLUMNS = ["tn", "fp", "fn", "tp"]
CLASS_NAMES = ["Retained", "Churned"]


def get_scores(model, X) -> tuple[np.ndarray, bool]:
    """Return (churn scores, scores_are_probabilities).

    Uses predict_proba when the model has it, otherwise decision_function
    (for example SVC with probability=False). Either works for ROC-AUC and
    PR-AUC, which only need a ranking.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1], True
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float).ravel(), False
    raise TypeError("Model has neither predict_proba nor decision_function.")


def compute_metrics(
    y_true, y_pred, y_score, score_is_probability: bool = True
) -> dict[str, float]:
    """All scalar metrics plus the four confusion-matrix counts.

    precision, recall, f1, specificity, balanced_accuracy and mcc depend on
    y_pred, which uses the model's default threshold. roc_auc and pr_auc
    depend only on the ranking in y_score. Brier needs real probabilities.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y_true)) == 2

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier": (
            float(brier_score_loss(y_true, y_score)) if score_is_probability else float("nan")
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def classification_report_text(y_true, y_pred) -> str:
    """Readable per-class precision, recall and F1."""
    return classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, digits=3, zero_division=0
    )