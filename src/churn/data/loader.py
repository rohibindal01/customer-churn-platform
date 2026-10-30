"""Load the raw Telco churn CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.config.settings import get_settings
from churn.data.fingerprint import short_fingerprint
from churn.data.validator import validate_or_raise
from churn.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(path: Path | str | None = None) -> pd.DataFrame:
    """Read the CSV exactly as stored, with no cleaning."""
    csv_path = Path(path) if path else get_settings().raw_data_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Run: python scripts\\01_download_data.py"
        )
    df = pd.read_csv(csv_path)
    logger.info(
        "Loaded %d rows x %d columns from %s (fingerprint %s)",
        df.shape[0], df.shape[1], csv_path.name, short_fingerprint(csv_path),
    )
    return df


def load_validated_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load the CSV and raise DataValidationError if it breaks the contract."""
    df = load_raw_data(path)
    validate_or_raise(df, require_target=True)
    return df