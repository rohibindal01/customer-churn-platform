"""Tests for the data contract and the file fingerprint."""
from churn.config.settings import get_settings
from churn.data import schema
from churn.data.fingerprint import file_sha256, short_fingerprint


def test_schema_has_21_raw_columns():
    assert len(schema.RAW_COLUMNS) == 21
    assert len(schema.RAW_FEATURE_COLUMNS) == 19


def test_feature_groups_partition_the_features():
    grouped = (
        set(schema.CATEGORICAL_COLUMNS)
        | set(schema.NUMERIC_COLUMNS)
        | {schema.SENIOR_COLUMN}
    )
    assert grouped == set(schema.RAW_FEATURE_COLUMNS)


def test_schema_matches_config():
    data_cfg = get_settings().data
    assert data_cfg.target_column == schema.TARGET_COLUMN
    assert data_cfg.id_column == schema.ID_COLUMN
    assert data_cfg.positive_label == schema.POSITIVE_LABEL


def test_fingerprint_is_stable(tmp_path):
    file_path = tmp_path / "sample.csv"
    file_path.write_text("a,b\n1,2\n", encoding="utf-8")
    assert file_sha256(file_path) == file_sha256(file_path)
    assert short_fingerprint(file_path) == file_sha256(file_path)[:12]


def test_fingerprint_changes_when_content_changes(tmp_path):
    file_path = tmp_path / "sample.csv"
    file_path.write_text("a,b\n1,2\n", encoding="utf-8")
    first = file_sha256(file_path)
    file_path.write_text("a,b\n1,3\n", encoding="utf-8")
    assert file_sha256(file_path) != first