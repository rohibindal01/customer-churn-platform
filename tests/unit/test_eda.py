"""Unit tests for the EDA helpers, using small synthetic data."""
import numpy as np
import pandas as pd
import pytest

from churn.data import eda


def _make_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    tenure = rng.integers(0, 73, size=n)
    tenure[:2] = 0
    monthly = np.round(rng.uniform(18, 118, size=n), 2)
    total = [" " if t == 0 else f"{t * m:.2f}" for t, m in zip(tenure, monthly)]
    services = ["Yes", "No", "No internet service"]
    return pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], size=n),
            "SeniorCitizen": rng.integers(0, 2, size=n),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "Dependents": rng.choice(["Yes", "No"], size=n),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], size=n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], size=n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
            "OnlineSecurity": rng.choice(services, size=n),
            "OnlineBackup": rng.choice(services, size=n),
            "DeviceProtection": rng.choice(services, size=n),
            "TechSupport": rng.choice(services, size=n),
            "StreamingTV": rng.choice(services, size=n),
            "StreamingMovies": rng.choice(services, size=n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
            "PaperlessBilling": rng.choice(["Yes", "No"], size=n),
            "PaymentMethod": rng.choice(["Electronic check", "Mailed check"], size=n),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": ["Yes" if i % 4 == 0 else "No" for i in range(n)],
        }
    )


def test_prepare_converts_blank_total_charges_and_flags_churn():
    df = _make_df()
    data = eda.prepare_eda_frame(df, "Churn", "Yes")
    assert data["TotalCharges"].isna().sum() == (df["tenure"] == 0).sum()
    assert pd.api.types.is_numeric_dtype(data["TotalCharges"])
    assert data[eda.CHURN_FLAG].sum() == 30
    assert set(data[eda.STATUS].unique()) == {"Churned", "Retained"}


def test_prepare_raises_on_missing_required_column():
    df = _make_df().drop(columns=["TotalCharges"])
    with pytest.raises(ValueError):
        eda.prepare_eda_frame(df, "Churn", "Yes")


def test_tenure_groups_use_expected_bands():
    df = pd.DataFrame(
        {
            "tenure": [0, 12, 13, 72],
            "MonthlyCharges": [20.0, 30.0, 40.0, 50.0],
            "TotalCharges": [" ", "360", "520", "3600"],
            "Churn": ["No", "Yes", "No", "No"],
        }
    )
    data = eda.prepare_eda_frame(df, "Churn", "Yes")
    assert list(data[eda.TENURE_GROUP].astype(str)) == ["0-12", "0-12", "13-24", "61+"]


def test_churn_rate_table_values():
    df = pd.DataFrame(
        {"Contract": ["A", "A", "B", "B"], eda.CHURN_FLAG: [1, 0, 0, 0]}
    )
    table = eda.churn_rate_table(df, "Contract").set_index("category")
    assert table.loc["A", "churn_rate_pct"] == 50.0
    assert table.loc["B", "churn_rate_pct"] == 0.0
    assert table.loc["A", "customers"] == 2


def test_cramers_v_detects_perfect_and_no_association():
    df = pd.DataFrame(
        {
            eda.CHURN_FLAG: [1, 0] * 20,
            "perfect": ["a", "b"] * 20,
            "noise": ["x", "x", "y", "y"] * 10,
        }
    )
    table = eda.categorical_association_table(df, ["perfect", "noise"]).set_index("feature")
    assert table.loc["perfect", "cramers_v"] == pytest.approx(1.0)
    assert table.loc["noise", "cramers_v"] == pytest.approx(0.0, abs=1e-9)


def test_numeric_association_sign_is_positive_when_churners_are_higher():
    df = pd.DataFrame(
        {
            eda.CHURN_FLAG: [1] * 20 + [0] * 20,
            "MonthlyCharges": list(range(100, 120)) + list(range(20)),
        }
    )
    table = eda.numeric_association_table(df, ["MonthlyCharges"])
    assert table.loc[0, "rank_biserial"] == pytest.approx(1.0)
    assert table.loc[0, "median_churned"] > table.loc[0, "median_retained"]


def test_iqr_outlier_table_flags_planted_outlier():
    df = pd.DataFrame({"x": list(range(1, 21)) + [1000]})
    table = eda.iqr_outlier_table(df, ["x"])
    assert table.loc[0, "outliers"] == 1


def test_run_eda_writes_expected_files(tmp_path):
    result = eda.run_eda(_make_df(), tmp_path, min_segment_size=5)
    names = {p.name for p in result["files"]}
    expected = {
        "churn_distribution.png",
        "numeric_distributions.png",
        "categorical_churn_rates.png",
        "tenure_group_churn.png",
        "correlation_heatmap.png",
        "contract_internet_heatmap.html",
        "tenure_vs_monthly_scatter.html",
        "categorical_association.csv",
        "eda_summary.json",
        "eda_report.md",
    }
    assert expected <= names
    for path in result["files"]:
        assert path.exists() and path.stat().st_size > 0
    assert result["summary"]["churned"] == 30