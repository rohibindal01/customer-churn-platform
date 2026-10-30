"""Tests for the stateless cleaning rules."""
import pandas as pd
import pytest

from churn.features.cleaning import clean_features, drop_duplicate_rows


def test_blank_total_charges_with_zero_tenure_becomes_zero(valid_df):
    df = valid_df.copy()
    df.loc[0, "tenure"] = 0
    df.loc[0, "TotalCharges"] = " "
    cleaned = clean_features(df)
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])


def test_unparseable_total_charges_with_history_stays_missing(valid_df):
    df = valid_df.copy()
    df.loc[1, "TotalCharges"] = "abc"
    assert pd.isna(clean_features(df).loc[1, "TotalCharges"])


def test_clean_does_not_modify_input(valid_df):
    before = valid_df.copy()
    clean_features(valid_df)
    pd.testing.assert_frame_equal(valid_df, before)


def test_clean_requires_numeric_source_columns(valid_df):
    with pytest.raises(ValueError):
        clean_features(valid_df.drop(columns=["tenure"]))


def test_drop_duplicate_rows_reports_count(valid_df):
    doubled = pd.concat([valid_df, valid_df.iloc[[0]]], ignore_index=True)
    out, removed = drop_duplicate_rows(doubled)
    assert removed == 1
    assert len(out) == 4