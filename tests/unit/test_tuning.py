"""Tests for hyperparameter tuning, using the synthetic 400-row dataset."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid, ParameterSampler

from churn.features.splitting import split_xy
from churn.models.pipeline import build_model_pipeline
from churn.models.tuning import (
    SHORTLIST,
    STRATEGY,
    format_tuning_report,
    load_baseline,
    plot_tuning_comparison,
    save_tuning,
    search_space,
    summarize_results,
    tune_model,
    tune_models,
)


@pytest.fixture
def xy(synthetic_df):
    return split_xy(synthetic_df)


def _quick_lr(X, y):
    return tune_model("logistic_regression", X, y, n_splits=3, seed=42,
                      space=("grid", {"model__C": [1.0]}))


def test_shortlist_and_search_spaces():
    assert SHORTLIST == ("logistic_regression", "adaboost", "gradient_boosting",
                         "catboost", "xgboost")
    for key in SHORTLIST:
        kind, space = search_space(key)
        assert kind in {"grid", "random"}
        assert space and all(name.startswith("model__") for name in space)


def test_logistic_regression_uses_grid_search():
    assert search_space("logistic_regression")[0] == "grid"


def test_model_without_search_space_raises():
    with pytest.raises(KeyError):
        search_space("knn")


@pytest.mark.parametrize("key", SHORTLIST)
def test_sampled_parameters_match_the_pipeline(key):
    kind, space = search_space(key)
    sampler = ParameterGrid(space) if kind == "grid" else ParameterSampler(space, n_iter=3, random_state=0)
    for params in list(sampler)[:3]:
        build_model_pipeline(key, STRATEGY, seed=42).set_params(**params)


def test_l1_ratio_controls_sparsity(xy):
    # Confirms the penalty settings really take effect on this scikit-learn version.
    X, y = xy
    _, grid = search_space("logistic_regression")
    fixed = {k: v[0] for k, v in grid.items()
             if k in ("model__solver", "model__max_iter", "model__penalty")}

    def zero_coefficients(l1_ratio: float) -> int:
        pipe = build_model_pipeline("logistic_regression", STRATEGY, seed=42)
        pipe.set_params(**fixed, model__C=0.01, model__l1_ratio=l1_ratio).fit(X, y)
        return int((pipe.named_steps["model"].coef_ == 0).sum())

    assert zero_coefficients(1.0) > zero_coefficients(0.0)


def test_random_search_returns_best_candidate(xy):
    X, y = xy
    result = tune_model("adaboost", X, y, n_iter=2, n_splits=3, seed=42)
    assert result.status == "ok"
    assert result.n_candidates == 2
    assert 0.0 <= result.cv_pr_auc <= 1.0 and 0.0 <= result.cv_roc_auc <= 1.0
    assert set(result.best_params) == set(search_space("adaboost")[1])
    assert len(result.best_pipeline.predict(X)) == len(X)


def test_grid_search_with_custom_space(xy):
    X, y = xy
    result = tune_model("logistic_regression", X, y, n_splits=3, seed=42,
                        space=("grid", {"model__C": [0.1, 1.0]}))
    assert result.n_candidates == 2
    assert result.best_params["model__C"] in (0.1, 1.0)
    assert result.refit_seconds > 0


def test_tune_models_records_failures(xy):
    X, y = xy
    results = tune_models(["not_a_model"], X, y, n_iter=1, n_splits=3, seed=42)
    assert results[0].status == "failed"
    assert "not_a_model" in results[0].reason


def test_summary_compares_with_baseline(xy):
    X, y = xy
    result = _quick_lr(X, y)
    baseline = pd.DataFrame({"model": ["logistic_regression"],
                             "roc_auc_mean": [0.5], "pr_auc_mean": [0.25]})
    row = summarize_results([result], baseline).iloc[0]
    assert row["delta_pr_auc"] == pytest.approx(row["cv_pr_auc"] - 0.25)
    assert row["delta_roc_auc"] == pytest.approx(row["cv_roc_auc"] - 0.5)
    assert row["overfit_gap"] == pytest.approx(row["train_pr_auc"] - row["cv_pr_auc"])


def test_load_baseline_missing_file_is_empty(tmp_path):
    assert load_baseline(tmp_path / "missing.csv").empty


def test_load_baseline_keeps_strategy_none(tmp_path):
    path = tmp_path / "summary.csv"
    pd.DataFrame({"model": ["a", "a"], "strategy": ["none", "smote"],
                  "roc_auc_mean": [0.8, 0.9], "pr_auc_mean": [0.6, 0.7]}).to_csv(path, index=False)
    assert list(load_baseline(path)["roc_auc_mean"]) == [0.8]


def test_save_tuning_writes_and_reloads(xy, tmp_path):
    X, y = xy
    result = _quick_lr(X, y)
    summary = summarize_results([result], pd.DataFrame())
    written = save_tuning([result], summary, tmp_path / "metrics", tmp_path / "models",
                          meta={"n_splits": 3})
    assert all(path.exists() and path.stat().st_size > 0 for path in written)
    params = json.loads((tmp_path / "metrics" / "best_params.json").read_text(encoding="utf-8"))
    assert params["logistic_regression"]["model__C"] == 1.0
    loaded = joblib.load(tmp_path / "models" / "logistic_regression.joblib")
    np.testing.assert_allclose(loaded.predict_proba(X), result.best_pipeline.predict_proba(X))


def test_report_and_plot(xy, tmp_path):
    X, y = xy
    ok = _quick_lr(X, y)
    failed = tune_models(["not_a_model"], X, y, n_splits=3, seed=42)[0]
    summary = summarize_results([ok, failed], pd.DataFrame())
    report = format_tuning_report(summary, [ok, failed], n_splits=3)
    assert "logistic_regression" in report
    assert "FAILED MODELS" in report and "not_a_model" in report
    path = plot_tuning_comparison(summary, tmp_path / "plot.png")
    assert path.exists() and path.stat().st_size > 0