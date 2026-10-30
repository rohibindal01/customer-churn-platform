"""Tests for the engineered features, using the 4-row valid_df fixture."""
import pytest

from churn.data import eda
from churn.features import engineering
from churn.features.cleaning import clean_features
from churn.features.engineering import engineer_features


def _engineered(df):
    return engineer_features(clean_features(df))


def test_addon_count(valid_df):
    assert list(_engineered(valid_df)["addon_count"]) == [1, 2, 2, 0]


def test_no_security_or_support_needs_internet(valid_df):
    # Row 3 has no internet service at all, so it must not be flagged.
    assert list(_engineered(valid_df)["no_security_or_support"]) == [1, 0, 0, 0]


def test_is_auto_payment(valid_df):
    assert list(_engineered(valid_df)["is_auto_payment"]) == [0, 0, 0, 1]


def test_average_spend_and_gap_for_billed_customers(valid_df):
    out = _engineered(valid_df)
    assert out["avg_monthly_spend"].iloc[0] == pytest.approx(29.85)
    assert out["avg_monthly_spend"].iloc[2] == pytest.approx(54.075)
    assert out["charge_gap"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert out["charge_gap"].iloc[2] == pytest.approx(-0.225, abs=1e-9)


def test_never_billed_customer_falls_back_to_current_charge(valid_df):
    df = valid_df.copy()
    df.loc[0, "tenure"] = 0
    df.loc[0, "TotalCharges"] = " "
    out = _engineered(df)
    assert out.loc[0, "avg_monthly_spend"] == pytest.approx(29.85)
    assert out.loc[0, "charge_gap"] == pytest.approx(0.0)
    assert out.loc[0, "tenure_group"] == "0-12"


def test_tenure_groups(valid_df):
    assert list(_engineered(valid_df)["tenure_group"]) == [
        "0-12", "25-48", "0-12", "25-48"
    ]


def test_missing_column_raises(valid_df):
    cleaned = clean_features(valid_df).drop(columns=["TechSupport"])
    with pytest.raises(ValueError):
        engineer_features(cleaned)


def test_tenure_bands_match_eda():
    assert engineering.TENURE_LABELS == eda._TENURE_LABELS
    assert list(engineering.TENURE_BINS) == list(eda._TENURE_BINS)