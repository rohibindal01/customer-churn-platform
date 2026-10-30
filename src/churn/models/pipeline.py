"""Assemble one complete, flat pipeline:
clean -> engineer -> preprocess -> (sampler) -> model.

imblearn's Pipeline does not accept another Pipeline as an intermediate
step, because it must see every sampler directly to skip it at prediction
time. So the three feature steps from Milestone 3 are placed directly in
the imblearn Pipeline. They are still fitted inside every cross-validation
fold, so nothing is learned from validation rows.
"""
from __future__ import annotations

import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline

from churn.config.settings import get_settings
from churn.features.preprocessing import build_feature_pipeline
from churn.models.imbalance import build_sampler, uses_class_weight, validate_strategy
from churn.models.registry import build_estimator, get_spec

FEATURE_STEP_NAMES = ("clean", "engineer", "preprocess")


class NotApplicableError(ValueError):
    """Raised when a model and strategy cannot be combined."""


def feature_steps() -> list[tuple[str, object]]:
    """Fresh, unfitted feature steps taken out of the Milestone 3 pipeline."""
    steps = list(build_feature_pipeline().steps)
    names = tuple(name for name, _ in steps)
    if names != FEATURE_STEP_NAMES:
        raise RuntimeError(
            f"Unexpected feature steps {names}; expected {FEATURE_STEP_NAMES}"
        )
    return steps


def compute_pos_weight(y) -> float:
    """Number of retained customers divided by number of churned customers."""
    labels = np.asarray(y).astype(int)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0:
        raise ValueError("Cannot compute a class weight: no churned examples.")
    return negatives / positives


def check_applicable(model_key: str, strategy: str) -> str | None:
    """Return the reason a combination is not applicable, or None if it is."""
    validate_strategy(strategy)
    spec = get_spec(model_key)
    if strategy == "class_weight" and not spec.supports_class_weight:
        return spec.class_weight_note
    return None


def build_model_pipeline(
    model_key: str,
    strategy: str,
    seed: int | None = None,
    pos_weight: float = 1.0,
) -> ImbPipeline:
    """Build an unfitted, flat pipeline for one model and one imbalance strategy."""
    reason = check_applicable(model_key, strategy)
    if reason:
        raise NotApplicableError(f"{model_key} + {strategy}: {reason}")

    seed = get_settings().random_seed if seed is None else seed
    steps = feature_steps()

    sampler = build_sampler(strategy, seed)
    if sampler is not None:
        steps.append(("sampler", sampler))

    estimator = build_estimator(
        model_key, seed, weighted=uses_class_weight(strategy), pos_weight=pos_weight
    )
    steps.append(("model", estimator))
    return ImbPipeline(steps)