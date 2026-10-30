"""Integration tests against the real downloaded dataset."""
import pytest

from churn.config.settings import get_settings
from churn.data import schema
from churn.data.loader import load_raw_data
from churn.data.validator import validate_raw_dataframe

pytestmark = pytest.mark.skipif(
    not get_settings().raw_data_path.exists(),
    reason="Raw dataset not downloaded yet (run scripts\\01_download_data.py)",
)


def test_real_file_has_expected_columns():
    df = load_raw_data()
    assert set(df.columns) == set(schema.RAW_COLUMNS)


def test_real_file_passes_validation():
    report = validate_raw_dataframe(load_raw_data())
    assert report.is_valid, report.summary()