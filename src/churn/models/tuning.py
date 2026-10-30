"""Stage 2: hyperparameter tuning of the shortlisted models.

Each search runs stratified k-fold CV on the TRAINING split only, using the
same folds as the Milestone 4 benchmark, so tuned and baseline scores are
directly comparable. The validation and test splits are not touched.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from scipy.stats import loguniform, uniform  # noqa: E402
from sklearn.base import BaseEstimator  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from churn.models.pipeline import build_model_pipeline  # noqa: E402
from churn.models.registry import REGISTRY, get_spec  # noqa: E402
from churn.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

SHORTLIST = ("logistic_regression", "adaboost", "gradient_boosting", "catboost", "xgboost")
STRATEGY = "none"  # chosen in Milestone 4B: balancing hurt calibration, not ranking
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
REFIT_METRIC = "pr_auc"


# ---------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------
def _sklearn_version() -> tuple[int, int]:
    major, minor = sklearn.__version__.split(".")[:2]
    return int(major), int(minor)


def _logistic_grid() -> dict:
    grid = {
        "model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        "model__l1_ratio": [0.0, 0.5, 1.0],   # 0 = L2, 1 = L1, 0.5 = elastic net
        "model__solver": ["saga"],            # the solver that supports all three
        "model__max_iter": [5000],
    }
    # Before scikit-learn 1.8, l1_ratio is only used with penalty="elasticnet".
    if _sklearn_version() < (1, 8):
        grid["model__penalty"] = ["elasticnet"]
    return grid


def _spaces() -> dict[str, tuple[str, dict]]:
    """(search type, parameter space) for every tunable model.

    Integer settings are explicit lists so they stay plain Python ints.
    """
    return {
        "logistic_regression": ("grid", _logistic_grid()),
        "adaboost": ("random", {
            "model__n_estimators": [50, 100, 200, 300, 400, 500],
            "model__learning_rate": loguniform(0.01, 1.0),
            "model__estimator": [DecisionTreeClassifier(max_depth=d) for d in (1, 2, 3)],
        }),
        "gradient_boosting": ("random", {
            "model__n_estimators": [100, 200, 300, 400, 600],
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_depth": [2, 3, 4, 5],
            "model__subsample": uniform(0.6, 0.4),          # 0.6 to 1.0
            "model__min_samples_leaf": [5, 10, 20, 30, 50],
            "model__max_features": [None, "sqrt"],
        }),
        "catboost": ("random", {
            "model__iterations": [200, 300, 400, 500, 600, 800],
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__depth": [3, 4, 5, 6, 7, 8],
            "model__l2_leaf_reg": loguniform(1.0, 30.0),
            "model__random_strength": uniform(0.0, 2.0),
        }),
        "xgboost": ("random", {
            "model__n_estimators": [100, 200, 300, 400, 600, 800],
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_depth": [2, 3, 4, 5, 6],
            "model__min_child_weight": loguniform(1.0, 20.0),
            "model__subsample": uniform(0.6, 0.4),          # 0.6 to 1.0
            "model__colsample_bytree": uniform(0.5, 0.5),   # 0.5 to 1.0
            "model__reg_lambda": loguniform(0.1, 20.0),
            "model__reg_alpha": loguniform(0.001, 5.0),
            "model__gamma": uniform(0.0, 2.0),
        }),
    }


def search_space(model_key: str) -> tuple[str, dict]:
    """Return ('grid' or 'random', parameter space) for a model."""
    spaces = _spaces()
    if model_key not in spaces:
        raise KeyError(f"No search space for '{model_key}'. Tunable models: {list(spaces)}")
    return spaces[model_key]


# ---------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------
@dataclass
class TuningResult:
    model: str
    status: str = "ok"
    reason: str = ""
    search_type: str = ""
    n_candidates: int = 0
    n_failed: int = 0
    best_params: dict = field(default_factory=dict)
    cv_pr_auc: float = float("nan")
    cv_pr_auc_std: float = float("nan")
    cv_roc_auc: float = float("nan")
    cv_roc_auc_std: float = float("nan")
    train_pr_auc: float = float("nan")
    mean_fit_seconds: float = float("nan")
    refit_seconds: float = float("nan")
    search_seconds: float = float("nan")
    cv_results: pd.DataFrame | None = None
    best_pipeline: object | None = None


def tune_model(
    model_key: str,
    X: pd.DataFrame,
    y,
    n_iter: int = 40,
    n_splits: int = 5,
    seed: int = 42,
    n_jobs: int = 1,
    space: tuple[str, dict] | None = None,
) -> TuningResult:
    """Search one model's hyperparameters with stratified CV. Raises on failure."""
    get_spec(model_key)  # KeyError for an unknown model
    search_type, param_space = space if space is not None else search_space(model_key)

    pipe = build_model_pipeline(model_key, STRATEGY, seed=seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    common = dict(
        scoring=SCORING, refit=REFIT_METRIC, cv=cv, n_jobs=n_jobs,
        return_train_score=True, error_score=np.nan,
    )
    if search_type == "grid":
        search = GridSearchCV(pipe, param_space, **common)
    elif search_type == "random":
        search = RandomizedSearchCV(pipe, param_space, n_iter=n_iter, random_state=seed, **common)
    else:
        raise ValueError(f"Unknown search type '{search_type}'")

    X = X.reset_index(drop=True)
    labels = np.asarray(y).astype(int)
    started = perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        search.fit(X, labels)
    elapsed = perf_counter() - started

    results = pd.DataFrame(search.cv_results_)
    best = search.best_index_

    def value(column: str) -> float:
        return float(results.loc[best, column])

    return TuningResult(
        model=model_key,
        search_type=search_type,
        n_candidates=int(len(results)),
        n_failed=int(results["mean_test_pr_auc"].isna().sum()),
        best_params=dict(search.best_params_),
        cv_pr_auc=value("mean_test_pr_auc"),
        cv_pr_auc_std=value("std_test_pr_auc"),
        cv_roc_auc=value("mean_test_roc_auc"),
        cv_roc_auc_std=value("std_test_roc_auc"),
        train_pr_auc=value("mean_train_pr_auc"),
        mean_fit_seconds=value("mean_fit_time"),
        refit_seconds=float(search.refit_time_),
        search_seconds=elapsed,
        cv_results=results,
        best_pipeline=search.best_estimator_,
    )


def tune_models(
    model_keys,
    X: pd.DataFrame,
    y,
    n_iter: int = 40,
    n_splits: int = 5,
    seed: int = 42,
    n_jobs: int = 1,
) -> list[TuningResult]:
    """Tune each model in turn. A failure is recorded, never raised."""
    keys = list(model_keys)
    results: list[TuningResult] = []
    for i, key in enumerate(keys, start=1):
        logger.info("[%d/%d] Tuning %s", i, len(keys), key)
        try:
            result = tune_model(key, X, y, n_iter=n_iter, n_splits=n_splits,
                                seed=seed, n_jobs=n_jobs)
            logger.info("[%d/%d] %s: CV PR-AUC %.4f, ROC-AUC %.4f (%d candidates, %.1f min)",
                        i, len(keys), key, result.cv_pr_auc, result.cv_roc_auc,
                        result.n_candidates, result.search_seconds / 60)
        except Exception as exc:  # noqa: BLE001 - record and continue
            result = TuningResult(model=key, status="failed",
                                  reason=f"{type(exc).__name__}: {str(exc)[:300]}")
            logger.error("[%d/%d] %s FAILED: %s", i, len(keys), key, result.reason)
        results.append(result)
    return results


# ---------------------------------------------------------------------
# Summary, saving, report, plot
# ---------------------------------------------------------------------
def _jsonable(value):
    if isinstance(value, BaseEstimator):
        return repr(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def jsonable_params(params: dict) -> dict:
    """Parameters converted to plain JSON-friendly values."""
    return {key: _jsonable(value) for key, value in params.items()}


def load_baseline(summary_path: Path | str) -> pd.DataFrame:
    """Milestone 4 results for strategy 'none' (empty if the file is missing)."""
    path = Path(summary_path)
    columns = ["model", "roc_auc_mean", "pr_auc_mean"]
    if not path.exists():
        logger.warning("Benchmark summary not found at %s; no baseline comparison.", path)
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    return df.loc[df["strategy"] == STRATEGY, columns].reset_index(drop=True)


def summarize_results(results: list[TuningResult], baseline: pd.DataFrame) -> pd.DataFrame:
    """One row per successfully tuned model, compared with its baseline."""
    rows = [
        {
            "model": r.model,
            "search_type": r.search_type,
            "n_candidates": r.n_candidates,
            "n_failed": r.n_failed,
            "cv_pr_auc": r.cv_pr_auc,
            "cv_pr_auc_std": r.cv_pr_auc_std,
            "cv_roc_auc": r.cv_roc_auc,
            "cv_roc_auc_std": r.cv_roc_auc_std,
            "train_pr_auc": r.train_pr_auc,
            "mean_fit_seconds": r.mean_fit_seconds,
            "refit_seconds": r.refit_seconds,
            "search_seconds": r.search_seconds,
            "best_params": json.dumps(jsonable_params(r.best_params), sort_keys=True),
        }
        for r in results if r.status == "ok"
    ]
    summary = pd.DataFrame(rows, columns=[
        "model", "search_type", "n_candidates", "n_failed", "cv_pr_auc", "cv_pr_auc_std",
        "cv_roc_auc", "cv_roc_auc_std", "train_pr_auc", "mean_fit_seconds",
        "refit_seconds", "search_seconds", "best_params",
    ])

    if baseline.empty:
        base = pd.DataFrame({
            "model": pd.Series(dtype=str),
            "baseline_roc_auc": pd.Series(dtype=float),
            "baseline_pr_auc": pd.Series(dtype=float),
        })
    else:
        base = baseline.rename(columns={"roc_auc_mean": "baseline_roc_auc",
                                        "pr_auc_mean": "baseline_pr_auc"})
    summary = summary.merge(base[["model", "baseline_roc_auc", "baseline_pr_auc"]],
                            on="model", how="left")
    numeric = ["cv_pr_auc", "cv_roc_auc", "train_pr_auc", "baseline_pr_auc", "baseline_roc_auc"]
    summary[numeric] = summary[numeric].astype(float)
    summary["delta_pr_auc"] = summary["cv_pr_auc"] - summary["baseline_pr_auc"]
    summary["delta_roc_auc"] = summary["cv_roc_auc"] - summary["baseline_roc_auc"]
    summary["overfit_gap"] = summary["train_pr_auc"] - summary["cv_pr_auc"]
    return summary.sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)


def save_tuning(
    results: list[TuningResult],
    summary: pd.DataFrame,
    metrics_dir: Path | str,
    models_dir: Path | str,
    meta: dict | None = None,
) -> list[Path]:
    """Write per-model CV tables, tuned pipelines, the summary and best parameters."""
    metrics_dir, models_dir = Path(metrics_dir), Path(models_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    best_params: dict[str, dict] = {}

    for r in results:
        if r.status != "ok":
            continue
        cv_path = metrics_dir / f"cv_results_{r.model}.csv"
        r.cv_results.to_csv(cv_path, index=False)
        model_path = models_dir / f"{r.model}.joblib"
        joblib.dump(r.best_pipeline, model_path)
        best_params[r.model] = jsonable_params(r.best_params)
        written += [cv_path, model_path]

    summary_path = metrics_dir / "tuning_summary.csv"
    summary.to_csv(summary_path, index=False)
    params_path = metrics_dir / "best_params.json"
    params_path.write_text(json.dumps(best_params, indent=2, sort_keys=True), encoding="utf-8")
    written += [summary_path, params_path]

    if meta is not None:
        meta_path = metrics_dir / "tuning_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        written.append(meta_path)
    return written


def format_tuning_report(summary: pd.DataFrame, results: list[TuningResult],
                         n_splits: int = 5) -> str:
    """Plain-text report for the console and tuning_report.txt."""
    lines: list[str] = []
    if summary.empty:
        lines.append("No successful tuning runs.")
    else:
        view = summary[[
            "model", "search_type", "n_candidates", "cv_pr_auc", "cv_pr_auc_std",
            "baseline_pr_auc", "delta_pr_auc", "cv_roc_auc", "baseline_roc_auc",
            "delta_roc_auc", "overfit_gap", "search_seconds",
        ]].copy()
        view["search_seconds"] = view["search_seconds"] / 60
        view = view.rename(columns={
            "search_type": "search", "n_candidates": "cands", "cv_pr_auc_std": "+-",
            "baseline_pr_auc": "base_pr_auc", "baseline_roc_auc": "base_roc_auc",
            "search_seconds": "minutes",
        }).round(4)
        lines += [
            f"TUNED MODELS ({n_splits}-fold CV on the training split, best candidate chosen by PR-AUC)",
            view.to_string(index=False),
            "",
            "delta       = tuned CV score minus the Milestone 4 baseline (strategy 'none', same folds)",
            "overfit_gap = training PR-AUC minus CV PR-AUC of the chosen candidate",
            "",
            "BEST PARAMETERS",
        ]
        lines += [f"  {row.model}: {row.best_params}" for row in summary.itertuples(index=False)]

    failed = [r for r in results if r.status != "ok"]
    if failed:
        lines += ["", "FAILED MODELS"]
        lines += [f"  {r.model}: {r.reason}" for r in failed]

    lines += [
        "",
        "Note: the best of many candidates, scored on the same folds, is slightly optimistic.",
        "The validation split in Milestone 6 gives the honest check.",
    ]
    return "\n".join(lines)


def plot_tuning_comparison(summary: pd.DataFrame, path: Path | str) -> Path:
    """Baseline vs tuned PR-AUC for each tuned model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = summary.sort_values("cv_pr_auc").reset_index(drop=True)
    labels = [REGISTRY[m].label if m in REGISTRY else m for m in data["model"]]
    positions = np.arange(len(data))
    height = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 0.7 * len(data) + 2))
    ax.barh(positions - height / 2, data["baseline_pr_auc"].fillna(0.0), height=height,
            color="#B0B0B0", label="Baseline (Milestone 4 defaults)")
    ax.barh(positions + height / 2, data["cv_pr_auc"], height=height,
            xerr=data["cv_pr_auc_std"], color="#4C72B0", label="Tuned (error bar = fold std)")
    ax.set_yticks(positions, labels)
    values = pd.concat([data["baseline_pr_auc"], data["cv_pr_auc"]]).dropna()
    if len(values):
        ax.set_xlim(left=max(0.0, float(values.min()) - 0.05))
    ax.set_xlabel("Mean PR-AUC (cross-validation on the training split)")
    ax.set_title("Baseline vs tuned PR-AUC")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path