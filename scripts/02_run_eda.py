"""Run the exploratory data analysis and save tables, figures and a report."""
from __future__ import annotations

import hashlib
import sys

import pandas as pd

from churn.config.settings import get_settings
from churn.data.eda import run_eda
from churn.utils.logger import get_logger

logger = get_logger("run_eda")


def main() -> int:
    settings = get_settings()
    settings.ensure_directories()

    path = settings.raw_data_path
    if not path.exists():
        logger.error("Raw data file not found: %s. Run scripts\\01_download_data.py first.", path)
        return 1

    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    df = pd.read_csv(path)
    logger.info("Loaded %d rows x %d columns from %s (fingerprint %s)",
                len(df), df.shape[1], path.name, fingerprint)

    result = run_eda(
        df,
        output_dir=settings.eda_dir,
        target=settings.data.target_column,
        positive_label=settings.data.positive_label,
        id_column=settings.data.id_column,
        source_name=path.name,
        fingerprint=fingerprint,
    )

    summary = result["summary"]
    logger.info("Churn rate: %.2f%% (%d of %d customers)",
                summary["churn_rate_pct"], summary["churned"], summary["rows"])
    for item in summary["top_categorical_drivers"]:
        logger.info("Top association: %-18s Cramer's V = %.3f", item["feature"], item["cramers_v"])
    for file_path in result["files"]:
        logger.info("Wrote %s", file_path)
    logger.info("EDA complete. Open %s", settings.eda_dir / "eda_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())