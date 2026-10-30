"""Probability calibration and probability-quality metrics."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

CALIBRATION_METHODS = ("uncalibrated", "sigmoid", "isotonic")


def build_calibrated(pipeline, method: str, seed: int = 42, n_splits: int = 5, n_jobs: int = 1):
    """Return an UNFITTED estimator for one calibration method.

    'uncalibrated' is a fresh copy of the pipeline. 'sigmoid' (Platt scaling)
    and 'isotonic' wrap it in CalibratedClassifierCV: the calibrator is fitted
    on out-of-fold predictions inside the training data, then one base
    pipeline is refitted on all training rows (ensemble=False).
    """
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Unknown calibration '{method}'. Choose from {CALIBRATION_METHODS}.")
    base = clone(pipeline)
    if method == "uncalibrated":
        return base
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return CalibratedClassifierCV(estimator=base, method=method, cv=cv,
                                  ensemble=False, n_jobs=n_jobs)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Weighted mean gap between predicted and observed churn rate in equal-width bins."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(p[in_bin].mean() - y[in_bin].mean())
    return float(ece)


def probability_metrics(y_true, y_prob) -> dict[str, float]:
    """Ranking metrics (ROC-AUC, PR-AUC) and probability-quality metrics."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
    }