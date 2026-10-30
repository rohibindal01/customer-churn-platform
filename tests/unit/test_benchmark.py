"""Tests for the benchmark runner, using the synthetic 400-row dataset."""
import json

import numpy as np
import pandas as pd
import pytest

from churn.features.splitting import split_xy
from churn.models.benchmark import (
    benchmark_models,
    best_per_model,
    run_cv,
    save_benchmark,
)
from churn.models.benchmark_plots import make_all_plots


@pytest.fixture
def xy(synthetic_df):
    return split_xy(synthetic_df)


def test_run_cv_ok(xy):
    X, y = xy
    run = run_cv("logistic_regression", "none", X, y, n_splits=3, seed=42)
    assert run["status"] == "ok"
    assert len(run["folds"]) == 3
    assert not np.isnan(run["oof_scores"]).any()
    assert set(np.unique(run["oof_pred"])) <= {0, 1}
    assert all(row["fit_seconds"] > 0 for row in run["folds"])


def test_run_cv_not_applicable(xy):
    X, y = xy
    run = run_cv("knn", "class_weight", X, y, n_splits=3, seed=42)
    assert run["status"] == "not_applicable"
    assert run["folds"] == [] and run["reason"]


def test_run_cv_records_failure_instead_of_raising(xy):
    X, y = xy
    run = run_cv("not_a_model", "none", X, y, n_splits=3, seed=42)
    assert run["status"] == "failed"
    assert "not_a_model" in run["reason"]


def test_benchmark_models_collects_everything(xy):
    X, y = xy
    result = benchmark_models(
        X, y, model_keys=["logistic_regression", "naive_bayes"],
        strategies=["none", "class_weight", "smote"], n_splits=3, seed=42,
    )
    assert len(result.summary) == 5
    assert len(result.folds) == 15
    assert len(result.skipped) == 1
    assert result.skipped.iloc[0]["model"] == "naive_bayes"
    assert {"roc_auc_mean", "roc_auc_std", "pr_auc_mean", "fit_seconds_mean"} <= set(
        result.summary.columns
    )
    assert len(result.oof_scores) == 5
    assert result.meta["runs_ok"] == 5 and result.meta["runs_not_applicable"] == 1


def test_save_benchmark_writes_files(xy, tmp_path):
    X, y = xy
    result = benchmark_models(X, y, model_keys=["logistic_regression"],
                              strategies=["none", "smote"], n_splits=3, seed=42)
    paths = save_benchmark(result, tmp_path)
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0
    assert "y_true" in np.load(paths["oof"]).files
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    assert meta["n_splits"] == 3


def test_make_all_plots(xy, tmp_path):
    X, y = xy
    result = benchmark_models(X, y, model_keys=["logistic_regression", "decision_tree"],
                              strategies=["none", "class_weight"], n_splits=3, seed=42)
    files = make_all_plots(result, tmp_path)
    assert len(files) == 7
    assert all(path.exists() and path.stat().st_size > 0 for path in files)


def test_best_per_model_picks_highest_metric():
    summary = pd.DataFrame(
        {"model": ["a", "a", "b"], "strategy": ["none", "smote", "none"],
         "pr_auc_mean": [0.5, 0.6, 0.4]}
    )
    best = best_per_model(summary)
    assert list(best["model"]) == ["a", "b"]
    assert best.iloc[0]["strategy"] == "smote"