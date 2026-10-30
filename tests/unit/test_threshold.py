"""Tests for threshold analysis and the business settings."""
import numpy as np
import pytest

from churn.config.settings import BusinessSettings, ThresholdSettings, get_settings
from churn.models.threshold import (
    POLICIES,
    THRESHOLD_GRID,
    breakeven_probability,
    net_benefit,
    select_thresholds,
    sweep_thresholds,
)

BUSINESS = BusinessSettings(customer_value=500.0, offer_cost=50.0, save_rate=0.30)


def test_threshold_grid():
    assert len(THRESHOLD_GRID) == 99
    assert THRESHOLD_GRID[0] == pytest.approx(0.01)
    assert THRESHOLD_GRID[-1] == pytest.approx(0.99)
    assert np.isclose(THRESHOLD_GRID, 0.5).any()


def test_net_benefit_formula():
    # 10 x 0.3 x 500 - (10 + 5) x 50 = 1500 - 750
    assert net_benefit(10, 5, BUSINESS) == pytest.approx(750.0)


def test_breakeven_probability():
    assert breakeven_probability(BUSINESS) == pytest.approx(50 / 150)


def test_sweep_known_values():
    sweep = sweep_thresholds([1, 1, 0, 0], [0.9, 0.6, 0.4, 0.1], BUSINESS,
                             thresholds=[0.5, 0.95])
    row = sweep.iloc[0]
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (2, 0, 0, 2)
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(1.0)
    assert row["f1"] == pytest.approx(1.0)
    assert row["flagged_pct"] == pytest.approx(50.0)
    assert row["net_benefit"] == pytest.approx(200.0)
    high = sweep.iloc[1]
    assert high["tp"] == 0 and np.isnan(high["precision"])
    assert high["f1"] == 0.0 and high["net_benefit"] == 0.0


def test_recall_never_increases_with_threshold():
    rng = np.random.default_rng(0)
    sweep = sweep_thresholds(rng.integers(0, 2, 500), rng.random(500), BUSINESS)
    assert (np.diff(sweep["recall"].to_numpy()) <= 1e-12).all()


def test_select_thresholds_policies():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.3 * y + 0.7 * rng.random(500), 0, 1)
    sweep = sweep_thresholds(y, p, BUSINESS)
    choices = select_thresholds(sweep, recall_target=0.8)
    assert tuple(choices["policy"]) == POLICIES
    by = choices.set_index("policy")
    assert by.loc["default_0.5", "threshold"] == pytest.approx(0.5)
    assert by.loc["recall_target", "recall"] >= 0.8
    assert by.loc["max_f1", "f1"] == pytest.approx(sweep["f1"].max())
    assert by.loc["max_net_benefit", "net_benefit"] == pytest.approx(sweep["net_benefit"].max())


def test_unreachable_recall_target_is_flagged():
    sweep = sweep_thresholds([1, 1, 0, 0], [0.9, 0.6, 0.4, 0.1], BUSINESS, thresholds=[0.95])
    row = select_thresholds(sweep, 0.8).set_index("policy").loc["recall_target"]
    assert "not reachable" in row["note"]


def test_business_and_threshold_settings_load():
    settings = get_settings()
    assert settings.business.customer_value > 0
    assert 0 < settings.business.save_rate <= 1
    assert 0 < settings.threshold.recall_target <= 1


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        BusinessSettings(save_rate=0.0)
    with pytest.raises(ValueError):
        BusinessSettings(offer_cost=-1.0)
    with pytest.raises(ValueError):
        ThresholdSettings(recall_target=1.5)