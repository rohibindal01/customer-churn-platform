"""Fixtures for the feature and split tests (tests\\unit only)."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """400 invented customers in the raw CSV format, with 25% churn."""
    n = 400
    rng = np.random.default_rng(1)
    tenure = rng.integers(0, 73, size=n)
    tenure[:3] = 0
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
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"], size=n),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": ["Yes" if i % 4 == 0 else "No" for i in range(n)],
        }
    )