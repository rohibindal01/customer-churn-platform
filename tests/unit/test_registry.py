"""Tests for the model registry."""
import pytest

from churn.models.registry import REGISTRY, build_estimator, get_spec, list_model_keys


def test_registry_has_13_models():
    assert len(list_model_keys()) == 13


def test_families_are_known():
    assert {spec.family for spec in REGISTRY.values()} == {"classical", "ensemble", "boosting"}


@pytest.mark.parametrize("key", list_model_keys())
def test_every_model_builds(key):
    plain = build_estimator(key, seed=42)
    weighted = build_estimator(key, seed=42, weighted=True, pos_weight=2.7)
    assert hasattr(plain, "fit") and hasattr(weighted, "fit")


def test_class_weight_support_flags():
    unsupported = {k for k, spec in REGISTRY.items() if not spec.supports_class_weight}
    assert unsupported == {"knn", "naive_bayes", "adaboost", "gradient_boosting"}


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        get_spec("not_a_model")