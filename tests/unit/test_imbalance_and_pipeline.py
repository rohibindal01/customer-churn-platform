"""Tests for imbalance strategies and the assembled model pipeline."""
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from churn.features.splitting import split_xy
from churn.models.evaluation import get_scores
from churn.models.imbalance import STRATEGIES, build_sampler
from churn.models.pipeline import (
    NotApplicableError,
    build_model_pipeline,
    check_applicable,
    compute_pos_weight,
)
from churn.models.registry import list_model_keys


def test_there_are_five_strategies():
    assert STRATEGIES == ("none", "class_weight", "random_over", "random_under", "smote")


def test_sampler_types():
    assert build_sampler("none", 42) is None
    assert build_sampler("class_weight", 42) is None
    assert isinstance(build_sampler("random_over", 42), RandomOverSampler)
    assert isinstance(build_sampler("random_under", 42), RandomUnderSampler)
    assert isinstance(build_sampler("smote", 42), SMOTE)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        build_sampler("magic", 42)


@pytest.mark.parametrize("model_key", ["knn", "naive_bayes", "adaboost", "gradient_boosting"])
def test_class_weight_not_applicable(model_key):
    assert check_applicable(model_key, "class_weight")
    with pytest.raises(NotApplicableError):
        build_model_pipeline(model_key, "class_weight")


def test_class_weight_applicable_for_logistic_regression():
    assert check_applicable("logistic_regression", "class_weight") is None


@pytest.mark.parametrize(
    "strategy, expected",
    [
        ("none", ["features", "model"]),
        ("class_weight", ["features", "model"]),
        ("random_over", ["features", "sampler", "model"]),
        ("random_under", ["features", "sampler", "model"]),
        ("smote", ["features", "sampler", "model"]),
    ],
)
def test_pipeline_step_order(strategy, expected):
    pipe = build_model_pipeline("logistic_regression", strategy)
    assert [name for name, _ in pipe.steps] == expected


def test_class_weight_reaches_the_model():
    weighted = build_model_pipeline("logistic_regression", "class_weight")
    plain = build_model_pipeline("logistic_regression", "none")
    assert weighted.named_steps["model"].class_weight == "balanced"
    assert plain.named_steps["model"].class_weight is None
    xgb = build_model_pipeline("xgboost", "class_weight", pos_weight=3.0)
    assert xgb.named_steps["model"].scale_pos_weight == 3.0


def test_sampler_is_skipped_at_prediction_time(synthetic_df):
    X, y = split_xy(synthetic_df)
    pipe = build_model_pipeline("logistic_regression", "random_under", seed=42)
    pipe.fit(X.iloc[:300], y.iloc[:300])
    assert len(pipe.predict(X.iloc[300:])) == 100


def test_pos_weight_is_negatives_over_positives():
    assert compute_pos_weight(pd.Series([0, 0, 0, 1])) == pytest.approx(3.0)


def test_pos_weight_needs_positives():
    with pytest.raises(ValueError):
        compute_pos_weight([0, 0, 0])


@pytest.mark.parametrize("model_key", list_model_keys())
def test_every_model_fits_and_scores(synthetic_df, model_key):
    X, y = split_xy(synthetic_df)
    pipe = build_model_pipeline(model_key, "none", seed=42)
    pipe.fit(X, y)
    scores, is_probability = get_scores(pipe, X)
    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all()
    assert is_probability == (model_key != "svm")
    assert set(np.unique(pipe.predict(X))) <= {0, 1}