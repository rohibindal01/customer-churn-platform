"""Permutation feature importance on held-out (validation) data.

Each raw input column is shuffled, which breaks its link with churn, and
the drop in PR-AUC is measured. It works for any model because it only
calls the model's predictions.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def compute_permutation_importance(
    estimator,
    X: pd.DataFrame,
    y,
    n_repeats: int = 20,
    seed: int = 42,
    n_jobs: int = 1,
    scoring: str = "average_precision",
) -> pd.DataFrame:
    """Mean and std of the score drop per raw column, most important first."""
    X = X.reset_index(drop=True)
    labels = np.asarray(y).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        result = permutation_importance(
            estimator, X, labels, scoring=scoring, n_repeats=n_repeats,
            random_state=seed, n_jobs=n_jobs,
        )
    table = pd.DataFrame({
        "feature": list(X.columns),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
        # fraction of repeats in which shuffling made the model worse
        "share_positive": (result.importances > 0).mean(axis=1),
    })
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)