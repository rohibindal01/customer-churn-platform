"""Tests for MLflow tracking, using a temporary SQLite database."""
import pytest
from mlflow.tracking import MlflowClient

from churn.utils.tracking import log_run


def _uri(tmp_path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def test_log_run_records_params_metrics_tags_and_artifacts(tmp_path):
    uri = _uri(tmp_path)
    note = tmp_path / "note.txt"
    note.write_text("hello", encoding="utf-8")
    run_id = log_run(
        uri, "unit-test", tmp_path / "artifacts", "lr-test",
        params={"model__C": 1.0, "model__penalty": None},
        metrics={"cv_pr_auc": 0.6, "missing": float("nan")},
        tags={"stage": "tuning"},
        artifacts=[note],
    )
    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.metrics["cv_pr_auc"] == pytest.approx(0.6)
    assert "missing" not in run.data.metrics
    assert run.data.params["model__C"] == "1.0"
    assert run.data.tags["stage"] == "tuning"
    assert [f.path for f in client.list_artifacts(run_id)] == ["note.txt"]


def test_experiment_is_created_once(tmp_path):
    uri = _uri(tmp_path)
    first = log_run(uri, "unit-test", tmp_path / "artifacts", "a", {}, {"m": 1.0}, {})
    second = log_run(uri, "unit-test", tmp_path / "artifacts", "b", {}, {"m": 2.0}, {})
    client = MlflowClient(tracking_uri=uri)
    assert client.get_run(first).info.experiment_id == client.get_run(second).info.experiment_id