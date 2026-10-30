"""Create the stratified split, fit the feature pipeline on train only,
and print evidence for each engineered feature (training data only)."""
from __future__ import annotations

import json
import sys

import pandas as pd

from churn.config.settings import get_settings
from churn.data import eda, schema
from churn.data.fingerprint import short_fingerprint
from churn.data.loader import load_validated_data
from churn.features.cleaning import clean_features, drop_duplicate_rows
from churn.features.engineering import (
    ENGINEERED_CATEGORICAL,
    ENGINEERED_FLAGS,
    ENGINEERED_NUMERIC,
    engineer_features,
)
from churn.features.preprocessing import build_feature_pipeline, get_feature_names
from churn.features.splitting import save_splits, split_xy, stratified_split
from churn.utils.logger import get_logger

logger = get_logger("prepare_data")


def _show(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(table.to_string(index=False))


def main() -> int:
    settings = get_settings()
    settings.ensure_directories()

    df = load_validated_data()
    df, removed = drop_duplicate_rows(df)
    train, val, test = stratified_split(df, settings)
    manifest = save_splits(
        {"train": train, "validation": val, "test": test},
        extra={
            "source_fingerprint": short_fingerprint(settings.raw_data_path),
            "duplicates_removed": removed,
        },
        settings=settings,
    )

    print("\n=== SPLITS ===")
    for name, info in manifest["splits"].items():
        print(f"{name:<11} rows={info['rows']:<6} churned={info['churned']:<5} "
              f"churn_rate={info['churn_rate_pct']}%")

    # Fit on TRAIN only, then transform the other splits with what was learned.
    X_train, y_train = split_xy(train, settings)
    X_val, _ = split_xy(val, settings)
    X_test, _ = split_xy(test, settings)

    pipeline = build_feature_pipeline()
    train_matrix = pipeline.fit_transform(X_train)
    val_matrix = pipeline.transform(X_val)
    test_matrix = pipeline.transform(X_test)
    names = get_feature_names(pipeline)

    print("\n=== FEATURE MATRIX ===")
    print(f"train {train_matrix.shape}, validation {val_matrix.shape}, "
          f"test {test_matrix.shape}")
    print(f"NaN values: train={int(pd.isna(train_matrix).sum())}, "
          f"validation={int(pd.isna(val_matrix).sum())}, "
          f"test={int(pd.isna(test_matrix).sum())}")

    names_path = settings.metrics_dir / "feature_names.json"
    names_path.write_text(json.dumps(names, indent=2), encoding="utf-8")
    logger.info("Wrote %d feature names to %s", len(names), names_path)

    # Evidence for the engineered features, computed on training data only.
    engineered = engineer_features(clean_features(X_train))
    engineered[eda.CHURN_FLAG] = y_train.to_numpy()

    rate_tables = pd.concat(
        [
            eda.churn_rate_table(engineered, c)
            for c in [*ENGINEERED_FLAGS, *ENGINEERED_CATEGORICAL, "addon_count"]
        ],
        ignore_index=True,
    )
    numeric_tests = eda.numeric_association_table(engineered, ENGINEERED_NUMERIC)
    corr = engineered[[*schema.NUMERIC_COLUMNS, *ENGINEERED_NUMERIC]].corr(
        method="spearman"
    ).round(2)

    _show("ENGINEERED FEATURES: churn rate by value (train)", rate_tables)
    _show("ENGINEERED NUMERIC FEATURES: churned vs retained (train)",
          numeric_tests.round(3))
    print("\n=== SPEARMAN CORRELATION (train) ===")
    print(corr)

    rate_tables.to_csv(settings.eda_dir / "engineered_feature_rates.csv", index=False)
    numeric_tests.to_csv(settings.eda_dir / "engineered_feature_tests.csv", index=False)

    logger.info("Data preparation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())