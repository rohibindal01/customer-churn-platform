"""Cross-validated benchmark of every model under every imbalance strategy."""
from __future__ import annotations

import json
import platform
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from churn.models.evaluation import (
    CONFUSION_COLUMNS,
    METRIC_COLUMNS,
    compute_metrics,
    get_scores,
)
from churn.models.imbalance import STRATEGIES
from churn.models.pipeline import build_model_pipeline, check_applicable, compute_pos_weight
from churn.models.registry import list_model_keys
from churn.utils.logger import get_logger

logger = get_logger(__name__)

_PACKAGES = ["scikit-learn", "imbalanced-learn", "xgboost", "lightgbm",
             "catboost", "pandas", "numpy"]


@dataclass
class BenchmarkResult:
    folds: pd.DataFrame          # one row per model, strategy and fold
    summary: pd.DataFrame        # mean and std over folds, one row per run
    skipped: pd.DataFrame        # not applicable or failed runs, with reasons
    oof_scores: dict             # "model__strategy" -> out-of-fold scores
    oof_pred: dict               # "model__strategy" -> out-of-fold 0/1 labels
    y_true: np.ndarray
    meta: dict


def run_cv(
    model_key: str,
    strategy: str,
    X: pd.DataFrame,
    y,
    n_splits: int,
    seed: int,
    position: str = "",
) -> dict:
    """Cross-validate one model with one strategy. Never raises.

    A failure is recorded in the returned dict, so one broken model cannot
    stop a 60-run benchmark.
    """
    run: dict = {
        "model": model_key, "strategy": strategy, "status": "ok", "reason": "",
        "folds": [], "oof_scores": None, "oof_pred": None, "seconds": 0.0,
    }
    started = perf_counter()
    try:
        reason = check_applicable(model_key, strategy)
        if reason:
            run.update(status="not_applicable", reason=reason)
            logger.info("[%s] %s + %s: not applicable", position, model_key, strategy)
            return run

        X = X.reset_index(drop=True)
        labels = np.asarray(y).astype(int)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_scores = np.full(len(labels), np.nan)
        oof_pred = np.full(len(labels), -1, dtype=np.int8)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=DeprecationWarning)
            for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, labels), start=1):
                pipe = build_model_pipeline(
                    model_key, strategy, seed=seed,
                    pos_weight=compute_pos_weight(labels[train_idx]),
                )
                t0 = perf_counter()
                pipe.fit(X.iloc[train_idx], labels[train_idx])
                fit_seconds = perf_counter() - t0

                X_valid = X.iloc[valid_idx]
                t0 = perf_counter()
                scores, is_probability = get_scores(pipe, X_valid)
                score_seconds = perf_counter() - t0
                predicted = np.asarray(pipe.predict(X_valid)).astype(int).ravel()

                oof_scores[valid_idx] = scores
                oof_pred[valid_idx] = predicted
                row = {
                    "model": model_key, "strategy": strategy, "fold": fold,
                    "n_train": int(len(train_idx)), "n_valid": int(len(valid_idx)),
                    "fit_seconds": fit_seconds,
                    "ms_per_1000_rows": score_seconds * 1000.0 / len(valid_idx) * 1000.0,
                    "score_type": "probability" if is_probability else "decision_function",
                }
                row.update(compute_metrics(labels[valid_idx], predicted, scores, is_probability))
                run["folds"].append(row)

        run["oof_scores"] = oof_scores
        run["oof_pred"] = oof_pred
        mean_auc = float(np.mean([r["roc_auc"] for r in run["folds"]]))
        run["seconds"] = perf_counter() - started
        logger.info("[%s] %s + %s: ROC-AUC %.4f in %.1fs",
                    position, model_key, strategy, mean_auc, run["seconds"])
    except Exception as exc:  # noqa: BLE001 - record any failure and carry on
        run.update(status="failed", reason=f"{type(exc).__name__}: {str(exc)[:300]}",
                   folds=[], oof_scores=None, oof_pred=None,
                   seconds=perf_counter() - started)
        logger.error("[%s] %s + %s FAILED: %s", position, model_key, strategy, run["reason"])
    return run


def summarize_folds(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over folds, sorted by ROC-AUC."""
    if folds.empty:
        return pd.DataFrame()
    metric_cols = [c for c in METRIC_COLUMNS if c in folds.columns]
    timing_cols = ["fit_seconds", "ms_per_1000_rows"]
    grouped = folds.groupby(["model", "strategy"], sort=False)
    parts = [
        grouped.size().rename("folds"),
        grouped[metric_cols + timing_cols].mean().add_suffix("_mean"),
        grouped[metric_cols].std(ddof=1).add_suffix("_std"),
        grouped["score_type"].first(),
    ]
    out = pd.concat(parts, axis=1).reset_index()
    return out.sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def best_per_model(summary: pd.DataFrame, metric: str = "pr_auc_mean") -> pd.DataFrame:
    """The strategy with the highest value of `metric` for each model."""
    ranked = summary.sort_values(metric, ascending=False)
    return ranked.drop_duplicates("model").reset_index(drop=True)


def _versions() -> dict:
    found = {}
    for name in _PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = "not installed"
    return found


def benchmark_models(
    X: pd.DataFrame,
    y,
    model_keys=None,
    strategies=None,
    n_splits: int = 5,
    seed: int = 42,
    n_jobs: int = 1,
    extra_meta: dict | None = None,
) -> BenchmarkResult:
    """Run every (model, strategy) pair with stratified cross-validation."""
    model_keys = list(model_keys) if model_keys else list_model_keys()
    strategies = list(strategies) if strategies else list(STRATEGIES)
    tasks = [(m, s) for m in model_keys for s in strategies]
    total = len(tasks)
    labels = np.asarray(y).astype(int)
    logger.info("Benchmark: %d runs x %d folds on %d rows (n_jobs=%d)",
                total, n_splits, len(labels), n_jobs)

    started = perf_counter()
    runs = Parallel(n_jobs=n_jobs)(
        delayed(run_cv)(m, s, X, labels, n_splits, seed, f"{i}/{total}")
        for i, (m, s) in enumerate(tasks, start=1)
    )
    elapsed = perf_counter() - started

    rows = [row for run in runs for row in run["folds"]]
    folds = pd.DataFrame(rows)
    summary = summarize_folds(folds)
    skipped = pd.DataFrame(
        [{"model": r["model"], "strategy": r["strategy"],
          "status": r["status"], "reason": r["reason"]}
         for r in runs if r["status"] != "ok"],
        columns=["model", "strategy", "status", "reason"],
    )
    oof_scores = {f"{r['model']}__{r['strategy']}": r["oof_scores"]
                  for r in runs if r["status"] == "ok"}
    oof_pred = {f"{r['model']}__{r['strategy']}": r["oof_pred"]
                for r in runs if r["status"] == "ok"}

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": seed,
        "n_splits": n_splits,
        "n_jobs": n_jobs,
        "rows": int(len(labels)),
        "churn_rate_pct": round(float(labels.mean() * 100), 2),
        "models": model_keys,
        "strategies": strategies,
        "runs_ok": sum(r["status"] == "ok" for r in runs),
        "runs_not_applicable": sum(r["status"] == "not_applicable" for r in runs),
        "runs_failed": sum(r["status"] == "failed" for r in runs),
        "elapsed_seconds": round(elapsed, 1),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": _versions(),
    }
    if extra_meta:
        meta.update(extra_meta)
    logger.info("Benchmark finished in %.1f min: %d ok, %d not applicable, %d failed",
                elapsed / 60, meta["runs_ok"], meta["runs_not_applicable"], meta["runs_failed"])
    return BenchmarkResult(folds, summary, skipped, oof_scores, oof_pred, labels, meta)


def save_benchmark(result: BenchmarkResult, out_dir: Path | str) -> dict[str, Path]:
    """Write all tables, the out-of-fold predictions and the metadata."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "folds": directory / "benchmark_folds.csv",
        "summary": directory / "benchmark_summary.csv",
        "skipped": directory / "benchmark_not_run.csv",
        "oof": directory / "benchmark_oof.npz",
        "meta": directory / "benchmark_meta.json",
    }
    result.folds.to_csv(paths["folds"], index=False)
    result.summary.to_csv(paths["summary"], index=False)
    result.skipped.to_csv(paths["skipped"], index=False)
    arrays = {"y_true": result.y_true.astype(np.int8)}
    arrays.update(result.oof_scores)
    arrays.update({f"{k}__pred": v for k, v in result.oof_pred.items()})
    np.savez_compressed(paths["oof"], **arrays)
    paths["meta"].write_text(json.dumps(result.meta, indent=2), encoding="utf-8")
    return paths


def format_report(result: BenchmarkResult, top_n: int = 15) -> str:
    """Plain-text ranking tables for the console and a saved .txt file."""
    summary = result.summary
    if summary.empty:
        return "No successful runs."
    rename = {
        "roc_auc_mean": "roc_auc", "roc_auc_std": "+-", "pr_auc_mean": "pr_auc",
        "recall_mean": "recall", "precision_mean": "precision", "f1_mean": "f1",
        "fit_seconds_mean": "fit_s",
    }
    cols = ["model", "strategy", *rename]
    top = summary[cols].rename(columns=rename).head(top_n).round(4)
    best = best_per_model(summary)[cols].rename(columns=rename).round(4)
    by_strategy = (
        summary.groupby("strategy")[["roc_auc_mean", "pr_auc_mean", "recall_mean",
                                     "precision_mean", "f1_mean"]]
        .mean().rename(columns=rename).round(4)
    )
    lines = [
        f"TOP {top_n} RUNS BY MEAN ROC-AUC (5-fold CV on the training split)",
        top.to_string(index=False),
        "",
        "BEST STRATEGY PER MODEL (chosen by mean PR-AUC)",
        best.to_string(index=False),
        "",
        "AVERAGE BY STRATEGY (only over models that ran it, so not a perfectly fair comparison)",
        by_strategy.to_string(),
        "",
        f"Runs: {result.meta['runs_ok']} ok, {result.meta['runs_not_applicable']} "
        f"not applicable, {result.meta['runs_failed']} failed. "
        f"Elapsed: {result.meta['elapsed_seconds']} seconds.",
    ]
    failed = result.skipped[result.skipped["status"] == "failed"]
    if not failed.empty:
        lines += ["", "FAILED RUNS"]
        lines += [f"  {r.model} + {r.strategy}: {r.reason}" for r in failed.itertuples()]
    return "\n".join(lines)