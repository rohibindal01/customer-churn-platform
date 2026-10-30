"""The five imbalance-handling strategies compared in the benchmark."""
from __future__ import annotations

from imblearn.base import BaseSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

STRATEGIES = ("none", "class_weight", "random_over", "random_under", "smote")

STRATEGY_LABELS = {
    "none": "No resampling",
    "class_weight": "Class weights",
    "random_over": "Random oversampling",
    "random_under": "Random undersampling",
    "smote": "SMOTE",
}


def validate_strategy(strategy: str) -> None:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}.")


def uses_class_weight(strategy: str) -> bool:
    """True when the model itself should weight the churn class."""
    validate_strategy(strategy)
    return strategy == "class_weight"


def build_sampler(strategy: str, seed: int) -> BaseSampler | None:
    """Return a sampler for resampling strategies, otherwise None.

    The default sampling_strategy ('auto') balances the classes 1:1. The
    sampler is placed inside an imblearn Pipeline, which applies it only
    while fitting, never when predicting.
    """
    validate_strategy(strategy)
    if strategy == "random_over":
        return RandomOverSampler(random_state=seed)
    if strategy == "random_under":
        return RandomUnderSampler(random_state=seed)
    if strategy == "smote":
        return SMOTE(k_neighbors=5, random_state=seed)
    return None