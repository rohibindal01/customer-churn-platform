"""Validation of raw Telco customer data (training file or uploaded batch)."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from churn.data import schema
from churn.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """Outcome of validating a dataframe."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [f"Valid: {self.is_valid}"]
        lines += [f"ERROR   - {message}" for message in self.errors]
        lines += [f"WARNING - {message}" for message in self.warnings]
        return "\n".join(lines)


class DataValidationError(ValueError):
    """Raised when data fails validation; carries the full report."""

    def __init__(self, report: ValidationReport) -> None:
        super().__init__("Data validation failed:\n" + report.summary())
        self.report = report


def _blank_mask(series: pd.Series) -> pd.Series:
    """True where a value is an empty or whitespace-only string."""
    text = series.astype("string").str.strip()
    return (text == "").fillna(False).astype(bool)


def _to_numeric(series: pd.Series) -> pd.Series:
    """Convert to float; blanks and unparseable text become NaN."""
    text = series.astype("string").str.strip()
    text = text.mask((text == "").fillna(False))
    converted = pd.to_numeric(text, errors="coerce")
    values = converted.to_numpy(dtype="float64", na_value=float("nan"))
    return pd.Series(values, index=series.index)


def _check_categoricals(df: pd.DataFrame, report: ValidationReport) -> None:
    for column, allowed in schema.CATEGORICAL_ALLOWED.items():
        if column not in df.columns:
            continue
        series = df[column]
        nulls = int(series.isna().sum())
        if nulls:
            report.warnings.append(f"Column '{column}' has {nulls} missing value(s).")
        bad = series[series.notna() & ~series.isin(list(allowed))]
        if not bad.empty:
            examples = bad.astype(str).value_counts().head(3).to_dict()
            report.errors.append(
                f"Column '{column}' has {len(bad)} value(s) outside "
                f"{list(allowed)}; examples: {examples}"
            )


def _check_numerics(df: pd.DataFrame, report: ValidationReport) -> None:
    for column in (*schema.NUMERIC_COLUMNS, schema.SENIOR_COLUMN):
        if column not in df.columns:
            continue
        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            blank = pd.Series(False, index=series.index)
            numeric = series.astype("float64")
        else:
            blank = _blank_mask(series)
            numeric = _to_numeric(series)

        n_null = int(series.isna().sum())
        n_blank = int(blank.sum())
        n_bad = int((numeric.isna() & series.notna() & ~blank).sum())
        report.stats[f"missing_{column}"] = n_null + n_blank

        if n_bad:
            report.errors.append(f"Column '{column}' has {n_bad} non-numeric value(s).")
        if n_blank:
            report.warnings.append(
                f"Column '{column}' has {n_blank} blank value(s); "
                "they will be treated as missing."
            )
        if n_null:
            report.warnings.append(f"Column '{column}' has {n_null} missing value(s).")

        negatives = int((numeric < 0).sum())
        if negatives:
            report.errors.append(f"Column '{column}' has {negatives} negative value(s).")

        if column == schema.SENIOR_COLUMN:
            present = numeric.dropna()
            invalid = int((~present.isin([0, 1])).sum())
            if invalid:
                report.errors.append(
                    f"Column '{column}' must be 0 or 1; found {invalid} other value(s)."
                )


def _check_target(df: pd.DataFrame, report: ValidationReport) -> None:
    series = df[schema.TARGET_COLUMN]
    allowed = [schema.NEGATIVE_LABEL, schema.POSITIVE_LABEL]

    nulls = int(series.isna().sum())
    if nulls:
        report.errors.append(f"Target column has {nulls} missing value(s).")
    bad = series[series.notna() & ~series.isin(allowed)]
    if not bad.empty:
        examples = bad.astype(str).value_counts().head(3).to_dict()
        report.errors.append(
            f"Target column has {len(bad)} value(s) outside {allowed}; examples: {examples}"
        )
    report.stats["churn_positive"] = int((series == schema.POSITIVE_LABEL).sum())


def _check_duplicates(df: pd.DataFrame, report: ValidationReport) -> None:
    if schema.ID_COLUMN in df.columns:
        duplicate_ids = int(df[schema.ID_COLUMN].duplicated().sum())
        if duplicate_ids:
            report.errors.append(f"{duplicate_ids} duplicated {schema.ID_COLUMN} value(s).")
    duplicate_rows = int(df.duplicated().sum())
    report.stats["duplicate_rows"] = duplicate_rows
    if duplicate_rows:
        report.warnings.append(f"{duplicate_rows} fully duplicated row(s).")


def validate_raw_dataframe(df: pd.DataFrame, require_target: bool = True) -> ValidationReport:
    """Check a raw dataframe against the data contract.

    Set require_target=False for inference data that has no Churn column.
    """
    report = ValidationReport()
    report.stats["rows"] = int(len(df))
    report.stats["columns"] = int(df.shape[1])

    if df.empty:
        report.errors.append("The dataset contains no rows.")
        return report

    required = list(schema.RAW_FEATURE_COLUMNS)
    if require_target:
        required.append(schema.TARGET_COLUMN)
    missing = [column for column in required if column not in df.columns]
    if missing:
        report.errors.append(f"Missing required columns: {missing}")

    extra = [column for column in df.columns if column not in schema.RAW_COLUMNS]
    if extra:
        report.warnings.append(f"Unexpected extra columns (will be ignored): {extra}")

    _check_categoricals(df, report)
    _check_numerics(df, report)
    if require_target and schema.TARGET_COLUMN in df.columns:
        _check_target(df, report)
    _check_duplicates(df, report)
    return report


def validate_or_raise(df: pd.DataFrame, require_target: bool = True) -> ValidationReport:
    """Validate and raise DataValidationError if any error is found."""
    report = validate_raw_dataframe(df, require_target=require_target)
    for message in report.warnings:
        logger.warning(message)
    if not report.is_valid:
        raise DataValidationError(report)
    return report