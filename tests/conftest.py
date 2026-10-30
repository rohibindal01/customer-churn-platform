"""Shared pytest fixtures."""
import pandas as pd
import pytest


@pytest.fixture
def valid_df() -> pd.DataFrame:
    """A tiny, schema-conformant sample in the raw CSV format."""
    return pd.DataFrame(
        {
            "customerID": ["0001-AAAA", "0002-BBBB", "0003-CCCC", "0004-DDDD"],
            "gender": ["Female", "Male", "Male", "Female"],
            "SeniorCitizen": [0, 0, 1, 0],
            "Partner": ["Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "No", "Yes"],
            "tenure": [1, 34, 2, 45],
            "PhoneService": ["No", "Yes", "Yes", "No"],
            "MultipleLines": ["No phone service", "No", "No", "No phone service"],
            "InternetService": ["DSL", "DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["No", "Yes", "Yes", "No internet service"],
            "OnlineBackup": ["Yes", "No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No", "No internet service"],
            "TechSupport": ["No", "No", "No", "No internet service"],
            "StreamingTV": ["No", "No", "No", "No internet service"],
            "StreamingMovies": ["No", "No", "No", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check", "Mailed check", "Mailed check",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
            "TotalCharges": ["29.85", "1889.5", "108.15", "1840.75"],
            "Churn": ["No", "No", "Yes", "No"],
        }
    )