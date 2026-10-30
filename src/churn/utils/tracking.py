"""Experiment tracking with MLflow, stored in a local SQLite file.

MlflowClient is created with an explicit tracking URI, so nothing depends
on global MLflow state and tests can point at a temporary database.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

from churn.utils.logger import get_logger

logger = get_logger(__name__)

_MAX_PARAM_LENGTH = 250


def _client(tracking_uri: str):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri)


def get_or_create_experiment(client, name: str, artifact_dir: Path) -> str:
    """Return the experiment id, creating the experiment the first time."""
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return client.create_experiment(name, artifact_location=artifact_dir.resolve().as_uri())


def log_run(
    tracking_uri: str,
    experiment_name: str,
    artifact_dir: Path | str,
    run_name: str,
    params: dict,
    metrics: dict,
    tags: dict,
    artifacts=(),
) -> str:
    """Log one finished run and return its run id. NaN metrics are skipped."""
    from mlflow.entities import Metric, Param

    client = _client(tracking_uri)
    experiment_id = get_or_create_experiment(client, experiment_name, Path(artifact_dir))
    run = client.create_run(
        experiment_id, run_name=run_name, tags={k: str(v) for k, v in tags.items()}
    )
    run_id = run.info.run_id
    now = int(time.time() * 1000)
    try:
        client.log_batch(
            run_id,
            metrics=[
                Metric(key, float(value), now, 0)
                for key, value in metrics.items()
                if value is not None and not math.isnan(float(value))
            ],
            params=[Param(key, str(value)[:_MAX_PARAM_LENGTH]) for key, value in params.items()],
        )
        for path in artifacts:
            client.log_artifact(run_id, str(path))
        client.set_terminated(run_id, "FINISHED")
    except Exception:
        client.set_terminated(run_id, "FAILED")
        raise
    return run_id