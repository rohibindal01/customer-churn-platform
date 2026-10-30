"""Smoke tests for configuration and logging."""
import logging

import pytest

from churn.config.settings import SplitSettings, get_settings
from churn.utils.logger import get_logger


def test_settings_load_from_yaml():
    settings = get_settings()
    assert settings.random_seed == 42
    assert settings.data.target_column == "Churn"
    assert settings.root.exists()


def test_split_sizes_sum_to_one():
    split = get_settings().split
    assert abs(split.train_size + split.validation_size + split.test_size - 1.0) < 1e-9


def test_invalid_split_is_rejected():
    with pytest.raises(ValueError):
        SplitSettings(train_size=0.7, validation_size=0.2, test_size=0.2)


def test_logger_uses_project_namespace():
    logger = get_logger("unit_test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "churn.unit_test"