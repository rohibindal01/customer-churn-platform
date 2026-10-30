"""Unit tests for raw-data validation."""
import pytest

from churn.data.validator import (
    DataValidationError,
    validate_or_raise,
    validate_raw_dataframe,
)


def test_valid_dataframe_passes(valid_df):
    report = validate_raw_dataframe(valid_df)
    assert report.is_valid
    assert report.stats["rows"] == 4


def test_empty_dataframe_fails(valid_df):
    assert not validate_raw_dataframe(valid_df.iloc[0:0]).is_valid


def test_missing_column_is_error(valid_df):
    report = validate_raw_dataframe(valid_df.drop(columns=["Contract"]))
    assert not report.is_valid
    assert any("Contract" in message for message in report.errors)


def test_target_is_optional_for_inference(valid_df):
    inference_df = valid_df.drop(columns=["Churn"])
    assert validate_raw_dataframe(inference_df, require_target=False).is_valid
    assert not validate_raw_dataframe(inference_df, require_target=True).is_valid


def test_unknown_category_is_error(valid_df):
    df = valid_df.copy()
    df.loc[0, "Contract"] = "Weekly"
    report = validate_raw_dataframe(df)
    assert not report.is_valid
    assert any("Contract" in message for message in report.errors)


def test_blank_total_charges_is_warning_not_error(valid_df):
    df = valid_df.copy()
    df.loc[0, "TotalCharges"] = " "
    report = validate_raw_dataframe(df)
    assert report.is_valid
    assert any("TotalCharges" in message for message in report.warnings)
    assert report.stats["missing_TotalCharges"] == 1


def test_non_numeric_total_charges_is_error(valid_df):
    df = valid_df.copy()
    df.loc[0, "TotalCharges"] = "abc"
    report = validate_raw_dataframe(df)
    assert not report.is_valid
    assert any("TotalCharges" in message for message in report.errors)


def test_negative_tenure_is_error(valid_df):
    df = valid_df.copy()
    df.loc[0, "tenure"] = -5
    assert not validate_raw_dataframe(df).is_valid


def test_senior_citizen_must_be_binary(valid_df):
    df = valid_df.copy()
    df.loc[0, "SeniorCitizen"] = 2
    assert not validate_raw_dataframe(df).is_valid


def test_duplicate_customer_id_is_error(valid_df):
    df = valid_df.copy()
    df.loc[1, "customerID"] = df.loc[0, "customerID"]
    report = validate_raw_dataframe(df)
    assert not report.is_valid
    assert any("customerID" in message for message in report.errors)


def test_invalid_target_value_is_error(valid_df):
    df = valid_df.copy()
    df.loc[0, "Churn"] = "Maybe"
    assert not validate_raw_dataframe(df).is_valid


def test_extra_columns_only_warn(valid_df):
    df = valid_df.copy()
    df["extra"] = 1
    report = validate_raw_dataframe(df)
    assert report.is_valid
    assert any("extra" in message for message in report.warnings)


def test_validate_or_raise_raises_with_report(valid_df):
    with pytest.raises(DataValidationError) as error_info:
        validate_or_raise(valid_df.drop(columns=["tenure"]))
    assert not error_info.value.report.is_valid