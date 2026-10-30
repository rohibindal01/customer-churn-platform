"""Unit tests for PredictionService, using a small model on synthetic data."""
import numpy as np
import pytest

from churn.analytics.segmentation import fit_segment_model
from churn.features.splitting import split_xy, stratified_split
from churn.models.pipeline import build_model_pipeline
from churn.serving.predictor import (
    BatchInputError,
    PredictionService,
    consistency_warnings,
    row_problems,
)

BANDS = {"Low", "Medium", "High", "Critical"}
META = {
    "model": "logistic_regression", "model_label": "Logistic Regression",
    "calibration": "uncalibrated",
    "threshold": {"value": 0.40, "policy": "max_net_benefit",
                  "all_policies": {"recall_target": 0.24, "max_f1": 0.40, "max_net_benefit": 0.40}},
    "test_metrics": {"confidence_intervals": {
        "roc_auc": {"value": 0.84, "ci_low": 0.82, "ci_high": 0.86}}},
}


@pytest.fixture
def parts(synthetic_df):
    train, _, _ = stratified_split(synthetic_df)
    X_train, y_train = split_xy(train)
    estimator = build_model_pipeline("logistic_regression", "none", seed=42).fit(X_train, y_train)
    segments = fit_segment_model(X_train, 3, seed=0)
    segments.names = {0: "S1", 1: "S2", 2: "S3"}
    return estimator, X_train, segments


@pytest.fixture
def service(parts):
    estimator, X_train, segments = parts
    return PredictionService(estimator, META, background=X_train.head(60), segment_model=segments)


def _features(df):
    return df.drop(columns=["Churn"])


def _record(df):
    return _features(df).drop(columns=["customerID"]).iloc[0].to_dict()


def test_valid_rows_have_no_problems(valid_df):
    assert (row_problems(valid_df) == "").all()


def test_row_problems_are_reported_per_row(valid_df):
    df = valid_df.astype(object).copy()
    df.loc[0, "Contract"] = "Weekly"
    df.loc[1, "tenure"] = "-5"
    df.loc[2, "MonthlyCharges"] = "abc"
    df.loc[3, "SeniorCitizen"] = 2
    problems = row_problems(df)
    assert "Contract has an unknown value" in problems[0]
    assert "tenure is negative" in problems[1]
    assert "MonthlyCharges is not a number" in problems[2]
    assert "SeniorCitizen must be 0 or 1" in problems[3]


def test_blank_total_charges_is_allowed(valid_df):
    df = valid_df.astype(object).copy()
    df.loc[0, "tenure"] = 0
    df.loc[0, "TotalCharges"] = " "
    assert row_problems(df)[0] == ""


def test_missing_column_rejects_the_whole_file(service, valid_df):
    with pytest.raises(BatchInputError) as info:
        service.predict_batch(valid_df.drop(columns=["Contract"]))
    assert "Contract" in str(info.value)


def test_batch_scores_valid_rows_and_rejects_invalid_rows(service, valid_df):
    df = _features(valid_df).astype(object)
    df.loc[1, "Contract"] = "Weekly"
    out = service.predict_batch(df)
    assert list(out["status"]) == ["scored", "rejected", "scored", "scored"]
    assert np.isnan(out.loc[1, "churn_probability"])
    scored = out[out["status"] == "scored"]
    assert scored["churn_probability"].between(0, 1).all()
    assert set(scored["risk_band"]) <= BANDS
    assert set(scored["segment"]) <= {"S1", "S2", "S3"}
    assert {"top_driver", "action_1", "action_2", "action_3"} <= set(out.columns)
    assert list(out["customerID"]) == list(valid_df["customerID"])


def test_whitespace_in_categories_is_normalised(service, valid_df):
    clean = service.predict_batch(_features(valid_df), explain=False)
    padded = _features(valid_df).astype(object)
    padded["Contract"] = "  " + padded["Contract"] + " "
    out = service.predict_batch(padded, explain=False)
    np.testing.assert_allclose(out["churn_probability"].to_numpy(dtype=float),
                               clean["churn_probability"].to_numpy(dtype=float))


def test_threshold_override_changes_the_decision(service, valid_df):
    # Check the rule itself, without assuming what the probabilities are:
    # predicted_churn is "Yes" exactly when probability >= threshold.
    features = _features(valid_df)
    prob = service.predict_batch(features, explain=False)["churn_probability"].to_numpy(dtype=float)

    for threshold in (0.05, 0.40, 0.90):
        out = service.predict_batch(features, threshold=threshold, explain=False)
        expected = np.where(prob >= threshold, "Yes", "No")
        assert list(out["predicted_churn"]) == list(expected)

    # Half of the smallest probability is below every probability, however tiny,
    # so everyone is flagged. (Synthetic rows can have probabilities near 1e-9.)
    if prob.min() > 0:
        everyone = service.predict_batch(features, threshold=prob.min() * 0.5, explain=False)
        assert (everyone["predicted_churn"] == "Yes").all()

    # Halfway between the largest probability and 0.95 is above every probability.
    if prob.max() < 0.95:
        above = prob.max() + (0.95 - prob.max()) * 0.5
        nobody = service.predict_batch(features, threshold=above, explain=False)
        assert (nobody["predicted_churn"] == "No").all()


def test_predict_one(service, valid_df):
    result = service.predict_one(_record(valid_df))
    assert 0 <= result.probability <= 1
    assert result.risk_band in BANDS and result.segment in {"S1", "S2", "S3"}
    assert result.base_value + result.drivers["shap"].sum() == pytest.approx(
        result.model_output, abs=1e-6)
    assert "actions" in result.recommendations and result.recommendations["disclaimer"]


def test_predict_one_rejects_invalid_input(service, valid_df):
    record = _record(valid_df)
    record["tenure"] = -1
    with pytest.raises(BatchInputError):
        service.predict_one(record)
    del record["Contract"]
    with pytest.raises(BatchInputError):
        service.predict_one(record)


def test_service_without_background_still_predicts(parts, valid_df):
    estimator, _, _ = parts
    service = PredictionService(estimator, META)
    assert not service.can_explain
    out = service.predict_batch(_features(valid_df))
    assert (out["status"] == "scored").all() and (out["top_driver"] == "").all()
    result = service.predict_one(_record(valid_df))
    assert result.drivers is None and result.segment is None
    assert "summary" in result.recommendations


def test_consistency_warnings():
    record = {"InternetService": "No", "OnlineSecurity": "Yes",
              "OnlineBackup": "No internet service", "DeviceProtection": "No internet service",
              "TechSupport": "No internet service", "StreamingTV": "No internet service",
              "StreamingMovies": "No internet service", "PhoneService": "No",
              "MultipleLines": "No", "tenure": 10, "MonthlyCharges": 50.0, "TotalCharges": 20.0}
    assert len(consistency_warnings(record)) == 3


def test_model_card_and_bands(service):
    card = service.model_card()
    assert card["model_label"] == "Logistic Regression"
    assert card["roc_auc"]["value"] == pytest.approx(0.84)
    assert service.bands_for(None).decision == pytest.approx(0.40)
    assert service.bands_for(0.30).decision == pytest.approx(0.30)