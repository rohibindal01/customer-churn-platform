"""Data contract for the IBM Telco Customer Churn dataset.

This is our *expectation* of the raw file. scripts/01b_inspect_data.py
compares it against the real download, and the validator enforces it.
"""
from __future__ import annotations

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"

SENIOR_COLUMN = "SeniorCitizen"  # stored as 0 / 1
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

_YES_NO = ("No", "Yes")
_SERVICE = ("No", "Yes", "No internet service")

CATEGORICAL_ALLOWED: dict[str, tuple[str, ...]] = {
    "gender": ("Female", "Male"),
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "MultipleLines": ("No", "Yes", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": _SERVICE,
    "OnlineBackup": _SERVICE,
    "DeviceProtection": _SERVICE,
    "TechSupport": _SERVICE,
    "StreamingTV": _SERVICE,
    "StreamingMovies": _SERVICE,
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}
CATEGORICAL_COLUMNS = list(CATEGORICAL_ALLOWED)

# Columns the model uses as inputs (everything except the ID and the target).
RAW_FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

RAW_COLUMNS = [ID_COLUMN, *RAW_FEATURE_COLUMNS, TARGET_COLUMN]