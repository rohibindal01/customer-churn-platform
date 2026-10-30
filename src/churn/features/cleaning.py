"""Stateless cleaning rules for raw Telco customer data.

Nothing here learns from the data, so these functions are safe to apply
to training data, validation data and a single customer at inference time.
"""
from __future__ import annotations

import pandas as pd

from churn.data import schema
from churn.utils.logger import get_logger

logger = get_logger(__name__)


def drop_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove fully duplicated rows and return (clean frame, rows removed)."""
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(out)
    if removed:
        logger.warning("Dropped %d fully duplicated row(s)", removed)
    return out, removed


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw feature frame.

    - tenure, MonthlyCharges and TotalCharges become float64. Blank or
      unparseable text becomes NaN.
    - A blank TotalCharges with tenure == 0 becomes 0.0: the customer has
      never been billed, so zero is the true value, not a guess.
    - Any other missing value is left as NaN for the imputer in the pipeline.
    """
    missing = [c for c in schema.NUMERIC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required numeric columns: {missing}")

    out = df.copy()
    for column in schema.NUMERIC_COLUMNS:
        text = out[column].astype("string").str.strip()
        out[column] = pd.to_numeric(text, errors="coerce").astype("float64")

    if schema.SENIOR_COLUMN in out.columns:
        out[schema.SENIOR_COLUMN] = pd.to_numeric(
            out[schema.SENIOR_COLUMN], errors="coerce"
        )

    never_billed = out["TotalCharges"].isna() & (out["tenure"] == 0)
    out.loc[never_billed, "TotalCharges"] = 0.0
    return out