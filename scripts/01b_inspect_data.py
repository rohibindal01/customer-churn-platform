"""Print a factual profile of the raw dataset to verify our assumptions."""
from __future__ import annotations

import pandas as pd

from churn.data import schema
from churn.data.loader import load_raw_data
from churn.data.validator import validate_raw_dataframe
from churn.utils.logger import get_logger

logger = get_logger("inspect_data")


def main() -> int:
    df = load_raw_data()
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 100)

    print("\n=== SHAPE ===")
    print(f"{df.shape[0]} rows x {df.shape[1]} columns")

    print("\n=== COLUMNS vs EXPECTED SCHEMA ===")
    missing = [c for c in schema.RAW_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in schema.RAW_COLUMNS]
    print(f"Missing from file : {missing}")
    print(f"Unexpected in file: {extra}")

    print("\n=== DTYPES, NULLS, UNIQUE VALUES ===")
    profile = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "nulls": df.isna().sum(),
            "unique": df.nunique(),
        }
    )
    print(profile)

    print("\n=== TotalCharges: blank / whitespace-only values ===")
    total = df["TotalCharges"]
    text = total.astype("string").str.strip()
    blank_mask = (text == "").fillna(False).astype(bool)
    print(f"Blank values: {int(blank_mask.sum())}")
    if blank_mask.any():
        print(f"Raw repr of first blank value: {total[blank_mask].iloc[0]!r}")
        print(df.loc[blank_mask, ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]])

    print("\n=== DUPLICATES ===")
    print(f"Fully duplicated rows : {int(df.duplicated().sum())}")
    print(f"Duplicated customerID : {int(df[schema.ID_COLUMN].duplicated().sum())}")

    print("\n=== CHURN DISTRIBUTION ===")
    counts = df[schema.TARGET_COLUMN].value_counts(dropna=False)
    shares = df[schema.TARGET_COLUMN].value_counts(normalize=True, dropna=False) * 100
    print(pd.DataFrame({"count": counts, "percent": shares.round(2)}))

    print("\n=== NUMERIC SUMMARY ===")
    numeric = pd.DataFrame(
        {
            "tenure": pd.to_numeric(df["tenure"], errors="coerce"),
            "MonthlyCharges": pd.to_numeric(df["MonthlyCharges"], errors="coerce"),
            "TotalCharges": pd.to_numeric(df["TotalCharges"], errors="coerce"),
        }
    )
    print(numeric.describe().round(2))
    print(f"SeniorCitizen values: {df['SeniorCitizen'].value_counts(dropna=False).to_dict()}")

    print("\n=== CATEGORICAL VALUES ===")
    for column in schema.CATEGORICAL_COLUMNS:
        print(f"{column}: {df[column].value_counts(dropna=False).to_dict()}")

    print("\n=== VALIDATION REPORT ===")
    report = validate_raw_dataframe(df)
    print(report.summary())
    print(f"Stats: {report.stats}")

    logger.info("Inspection complete (valid=%s)", report.is_valid)
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())