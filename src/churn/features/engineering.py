"""Feature engineering rules for the Telco churn data.

Input must already be cleaned (see cleaning.clean_features). These rules
are stateless: they use only the values in each row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ADDON_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
AUTO_PAYMENT_METHODS = ("Bank transfer (automatic)", "Credit card (automatic)")

# Keep in sync with the bands used in data/eda.py (a unit test checks this).
TENURE_BINS = [-1, 12, 24, 48, 60, np.inf]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-60", "61+"]

ENGINEERED_NUMERIC = ["addon_count", "avg_monthly_spend", "charge_gap"]
ENGINEERED_FLAGS = ["no_security_or_support", "is_auto_payment"]
ENGINEERED_CATEGORICAL = ["tenure_group"]

_REQUIRED = [
    *ADDON_COLUMNS, "InternetService", "PaymentMethod",
    "tenure", "MonthlyCharges", "TotalCharges",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the engineered columns added."""
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns needed for feature engineering: {missing}")

    out = df.copy()

    # How many add-on services the customer pays for.
    out["addon_count"] = (out[ADDON_COLUMNS] == "Yes").sum(axis=1).astype(int)

    # Has internet but neither security nor technical support.
    out["no_security_or_support"] = (
        (out["InternetService"] != "No")
        & (out["OnlineSecurity"] == "No")
        & (out["TechSupport"] == "No")
    ).astype(int)

    # Historical average spend per month. A customer with tenure 0 has no
    # history, so we use the current monthly charge.
    tenure = out["tenure"]
    months = tenure.where(tenure > 0)
    historical = out["TotalCharges"] / months
    out["avg_monthly_spend"] = historical.where(tenure > 0, out["MonthlyCharges"])

    # Positive: the customer now pays more than they have on average so far.
    out["charge_gap"] = out["MonthlyCharges"] - out["avg_monthly_spend"]

    out["is_auto_payment"] = out["PaymentMethod"].isin(AUTO_PAYMENT_METHODS).astype(int)

    bands = pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS)
    out["tenure_group"] = bands.astype(object)
    return out