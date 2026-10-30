"""Stage 2: tune the shortlisted models on the TRAINING split.

The validation and test splits are not touched here.

Examples (CMD):
    python scripts\\05_tune_models.py
    python scripts\\05_tune_models.py --models logistic_regression,adaboost --n-iter 3
    python scripts\\05_tune_models.py --n-jobs 4
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from importlib import metadata
from time import perf_counter

from churn.config.settings import get_settings
from churn.data.fingerprint import short_fingerprint
from churn.features.splitting import load_split, split_xy
from churn.models.tuning import (
    REFIT_METRIC,
    SCORING,
    SHORTLIST,
    STRATEGY,
    format_tuning_report,
    jsonable_params,
    load_baseline,
    plot_tuning_comparison,
    save_tuning,
    summarize_results,
    tune_models,
)
from churn.utils.logger import get_logger

logger = get_logger("tune_models")

_PACKAGES = ["scikit-learn", "imbalanced-learn", "xgboost", "catboost",
             "pandas", "numpy", "mlflow"]


def _versions() -> dict:
    found = {}
    for name in _PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = "not installed"
    return found


def _parse_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _log_to_mlflow(results, settings, out_dir, models_dir, fingerprint) -> None:
    from churn.utils.tracking import log_run

    for r in results:
        if r.status != "ok":
            continue
        try:
            run_id = log_run(
                tracking_uri=settings.mlflow_tracking_uri,
                experiment_name=settings.mlflow.experiment_name,
                artifact_dir=settings.root / "mlartifacts",
                run_name=f"tune_{r.model}",
                params={
                    **jsonable_params(r.best_params),
                    "search_type": r.search_type,
                    "n_candidates": r.n_candidates,
                    "strategy": STRATEGY,
                },
                metrics={
                    "cv_pr_auc": r.cv_pr_auc,
                    "cv_pr_auc_std": r.cv_pr_auc_std,
                    "cv_roc_auc": r.cv_roc_auc,
                    "cv_roc_auc_std": r.cv_roc_auc_std,
                    "train_pr_auc": r.train_pr_auc,
                    "mean_fit_seconds": r.mean_fit_seconds,
                    "refit_seconds": r.refit_seconds,
                    "search_seconds": r.search_seconds,
                },
                tags={
                    "stage": "tuning",
                    "model": r.model,
                    "split": "train",
                    "data_fingerprint": fingerprint,
                    "refit_metric": REFIT_METRIC,
                    "model_artifact": str(models_dir / f"{r.model}.joblib"),
                },
                artifacts=[out_dir / f"cv_results_{r.model}.csv"],
            )
            logger.info("MLflow run %s logged for %s", run_id, r.model)
        except Exception as exc:  # noqa: BLE001 - tracking must never break tuning
            logger.warning("MLflow logging failed for %s: %s", r.model, exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", help=f"Comma-separated models (default: {','.join(SHORTLIST)}).")
    parser.add_argument("--n-iter", type=int, default=40,
                        help="Candidates per randomized search (default 40). Grid search ignores it.")
    parser.add_argument("--folds", type=int, help="CV folds (default: cv.n_splits in config.yaml).")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel fits inside each search (default 1).")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    args = parser.parse_args()

    models = _parse_list(args.models) or list(SHORTLIST)
    unknown = [m for m in models if m not in SHORTLIST]
    if unknown:
        logger.error("No search space for %s. Tunable models: %s", unknown, list(SHORTLIST))
        return 1

    settings = get_settings()
    settings.ensure_directories()
    train = load_split("train", settings=settings)
    X_train, y_train = split_xy(train, settings)
    n_splits = args.folds or settings.cv_splits
    logger.info("Training split: %d rows, churn rate %.2f%%. Tuning %s with %d-fold CV.",
                len(X_train), y_train.mean() * 100, models, n_splits)

    started = perf_counter()
    results = tune_models(models, X_train, y_train, n_iter=args.n_iter, n_splits=n_splits,
                          seed=settings.random_seed, n_jobs=args.n_jobs)
    elapsed = perf_counter() - started

    baseline = load_baseline(settings.metrics_dir / "benchmark" / "benchmark_summary.csv")
    summary = summarize_results(results, baseline)

    fingerprint = short_fingerprint(settings.raw_data_path)
    out_dir = settings.metrics_dir / "tuning"
    models_dir = settings.models_dir / "tuned"
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": settings.random_seed,
        "n_splits": n_splits,
        "n_iter": args.n_iter,
        "n_jobs": args.n_jobs,
        "strategy": STRATEGY,
        "scoring": SCORING,
        "refit_metric": REFIT_METRIC,
        "models": models,
        "models_ok": sum(r.status == "ok" for r in results),
        "models_failed": sum(r.status != "ok" for r in results),
        "elapsed_seconds": round(elapsed, 1),
        "source_fingerprint": fingerprint,
        "split": "train",
        "python": sys.version.split()[0],
        "versions": _versions(),
    }
    files = save_tuning(results, summary, out_dir, models_dir, meta=meta)
    if not summary.empty:
        files.append(plot_tuning_comparison(summary,
                                            settings.plots_dir / "tuning" / "tuning_comparison.png"))

    report = format_tuning_report(summary, results, n_splits=n_splits)
    report_path = out_dir / "tuning_report.txt"
    report_path.write_text(report, encoding="utf-8")
    files.append(report_path)
    print("\n" + report)

    if args.no_mlflow:
        logger.info("MLflow logging skipped (--no-mlflow)")
    else:
        _log_to_mlflow(results, settings, out_dir, models_dir, fingerprint)

    for path in files:
        logger.info("Wrote %s", path)
    logger.info("Tuning finished in %.1f min", elapsed / 60)
    return 0 if any(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())