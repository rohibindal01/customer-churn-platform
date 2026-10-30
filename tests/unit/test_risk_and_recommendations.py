"""Tests for risk bands and the recommendation engine."""
import numpy as np
import pandas as pd
import pytest

from churn.analytics.plots import plot_risk_bands, plot_rule_frequency
from churn.analytics.recommendations import (
    BASES,
    DRIVER_FAMILIES,
    RULES,
    RuleConfig,
    customer_records,
    family_contributions,
    recommend,
)
from churn.analytics.risk import (
    RISK_BANDS,
    BandThresholds,
    adjust_decision,
    assign_risk_band,
    band_summary,
    bands_from_metadata,
)
from churn.features.splitting import split_xy

THRESHOLDS = BandThresholds(watch=0.24, decision=0.40, critical=0.70)
CONFIG = RuleConfig(high_spend_monthly=90.0)
CUSTOMER = {"Contract": "Month-to-month", "tenure": 3.0, "InternetService": "Fiber optic",
            "no_security_or_support": 1, "MonthlyCharges": 95.0,
            "PaymentMethod": "Electronic check", "charge_gap": 0.0, "SeniorCitizen": 0}


def test_assign_risk_band():
    bands = assign_risk_band([0.1, 0.3, 0.5, 0.8, 0.4, 0.7], THRESHOLDS)
    assert list(bands) == ["Low", "Medium", "High", "Critical", "High", "Critical"]


def test_invalid_thresholds_are_rejected():
    with pytest.raises(ValueError):
        BandThresholds(watch=0.5, decision=0.4)


def test_bands_from_metadata_and_adjustment():
    meta = {"threshold": {"value": 0.40, "all_policies": {"recall_target": 0.24}}}
    t = bands_from_metadata(meta)
    assert (t.watch, t.decision, t.critical) == (0.24, 0.40, 0.70)
    lowered = adjust_decision(t, 0.20)
    assert lowered.watch == 0.20 and lowered.decision == 0.20
    raised = adjust_decision(t, 0.80)
    assert raised.critical > 0.80


def test_band_summary():
    summary = band_summary([0.1, 0.3, 0.5, 0.8], ["Low", "Medium", "High", "Critical"],
                           [0, 0, 1, 1])
    assert list(summary["band"]) == list(RISK_BANDS)
    assert summary["customers"].sum() == 4
    assert summary.set_index("band").loc["Critical", "observed_churn_pct"] == 100.0


def test_family_contributions():
    groups = ["tenure", "tenure_group", "Contract", "PaymentMethod", "unknown_feature"]
    families = family_contributions(np.array([[1.0, -0.5, 0.2, 0.1, 0.3]]), groups)
    assert families.loc[0, "Tenure"] == pytest.approx(0.5)
    assert families.loc[0, "Other"] == pytest.approx(0.3)
    assert families.loc[0].sum() == pytest.approx(1.1)


def test_low_band_gets_no_actions():
    rec = recommend(CUSTOMER, 0.1, "Low", None, CONFIG)
    assert rec["actions"] == [] and "no proactive" in rec["summary"].lower()


def test_actions_follow_this_customers_drivers():
    contributions = {"Tenure": 1.9, "Internet service": 0.77, "Contract": 0.7,
                     "Security and support": 0.36, "Spend": -0.3}
    rec = recommend(CUSTOMER, 0.85, "Critical", contributions, CONFIG, max_items=10)
    ids = [a["rule_id"] for a in rec["actions"]]
    assert ids[:4] == ["onboarding", "fiber_service_review", "contract_incentive",
                       "security_support_trial"]
    assert {"high_value_loyalty", "autopay_nudge"} <= set(ids)
    assert "bill_increase_review" not in ids and "senior_assist" not in ids
    assert rec["disclaimer"]


def test_medium_band_skips_high_only_rules():
    ids = [a["rule_id"] for a in recommend(CUSTOMER, 0.3, "Medium", None, CONFIG, 10)["actions"]]
    assert "onboarding" in ids and "contract_incentive" not in ids


def test_missing_fields_do_not_crash():
    rec = recommend({"Contract": "Month-to-month"}, 0.8, "Critical", None, CONFIG)
    assert [a["rule_id"] for a in rec["actions"]] == ["contract_incentive"]
    with pytest.raises(ValueError):
        recommend(CUSTOMER, 0.5, "Extreme")


def test_rules_are_well_formed(synthetic_df):
    assert len({r.rule_id for r in RULES}) == len(RULES)
    for rule in RULES:
        assert rule.family in DRIVER_FAMILIES and rule.basis in BASES
        assert set(rule.bands) <= set(RISK_BANDS)
    X, _ = split_xy(synthetic_df)
    records = customer_records(X.head(3))
    assert len(records) == 3
    assert {"charge_gap", "no_security_or_support", "Contract"} <= set(records[0])


def test_recommendation_plots(tmp_path):
    summary = band_summary([0.1, 0.3, 0.5, 0.8], ["Low", "Medium", "High", "Critical"], [0, 0, 1, 1])
    paths = [plot_risk_bands(summary, tmp_path / "bands.png"),
             plot_rule_frequency(pd.Series({"A": 3, "B": 1}), tmp_path / "rules.png")]
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)