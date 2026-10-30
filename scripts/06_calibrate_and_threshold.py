"""Milestone 6A: calibrate the tuned models, compare them on the VALIDATION
split, analyse thresholds, and draw learning curves.

The test split is NOT loaded here.

Examples (CMD):
    python scripts\\06_calibrate_and_threshold.py --n-jobs 8
    python scripts\\06_calibrate_and_threshold.py --models logistic_regression --no-learning-curves
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter

from churn.config.settings import get_settings
from churn.data.fingerprint import short_fingerprint
from churn.features.splitting import load_split, split_xy
from churn.models.calibration import CALIBRATION_METHODS
from churn.models.calibration_plots import (
    plot_learning_curves,
    plot_reliability,
    plot_threshold_curves,
)
from churn.models.candidates import (
    best_calibration_per_model,
    comparison_table,
    compute_learning_curve,
    evaluate_candidates,
    format_candidates_report,
    learning_curve_table,
    load_tuned_pipelines,
    save_candidates,
    threshold_tables,
)
from churn.models.registry import REGISTRY
from churn.utils.logger import get_logger

logger = get_logger("calibrate_and_threshold")


def _parse_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _log_to_mlflow(results, choices, settings, fingerprint) -> None:
    from churn.utils.tracking import log_run

    business = settings.business
    for r in results:
        if r.status != "ok":
            continue
        try:
            picked = choices[(choices["model"] == r.model)
                             & (choices["calibration"] == r.calibration)].set_index("policy")
            metrics = {f"val_{k}": v for k, v in r.metrics.items()}
            metrics["fit_seconds"] = r.fit_seconds
            for policy in ("max_f1", "recall_target", "max_net_benefit"):
                metrics[f"threshold_{policy}"] = float(picked.loc[policy, "threshold"])
            metrics["val_net_benefit_per_1000_at_max"] = float(
                picked.loc["max_net_benefit", "net_benefit_per_1000"])
            run_id = log_run(
                tracking_uri=settings.mlflow_tracking_uri,
                experiment_name=settings.mlflow.experiment_name,
                artifact_dir=settings.root / "mlartifacts",
                run_name=f"calibrate_{r.key}",
                params={
                    "calibration": r.calibration,
                    "cv_folds": settings.cv_splits,
                    "recall_target": settings.threshold.recall_target,
                    "customer_value": business.customer_value,
                    "offer_cost": business.offer_cost,
                    "save_rate": business.save_rate,
                },
                metrics=metrics,
                tags={
                    "stage": "calibration",
                    "model": r.model,
                    "split": "validation",
                    "data_fingerprint": fingerprint,
                    "business_assumptions": "placeholder",
                    "model_artifact": str(settings.candidates_dir / f"{r.key}.joblib"),
                },
            )
            logger.info("MLflow run %s logged for %s", run_id, r.key)
        except Exception as exc:  # noqa: BLE001 - tracking must never break the analysis
            logger.warning("MLflow logging failed for %s: %s", r.key, exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", help="Comma-separated tuned models (default: all found).")
    parser.add_argument("--methods",
                        help=f"Comma-separated calibration methods (default: {','.join(CALIBRATION_METHODS)}).")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel fits inside calibration and learning curves (default 1).")
    parser.add_argument("--no-learning-curves", action="store_true",
                        help="Skip the learning curves (faster).")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_directories()

    tuned = load_tuned_pipelines(settings.tuned_models_dir)
    wanted = _parse_list(args.models)
    if wanted:
        unknown = [m for m in wanted if m not in tuned]
        if unknown:
            logger.error("No tuned model for %s. Available: %s", unknown, list(tuned))
            return 1
        tuned = {m: tuned[m] for m in wanted}
    methods = _parse_list(args.methods) or list(CALIBRATION_METHODS)
    bad = [m for m in methods if m not in CALIBRATION_METHODS]
    if bad:
        logger.error("Unknown calibration methods %s. Choose from %s", bad, list(CALIBRATION_METHODS))
        return 1

    X_train, y_train = split_xy(load_split("train", settings=settings), settings)
    X_val, y_val = split_xy(load_split("validation", settings=settings), settings)
    logger.info("Train %d rows, validation %d rows. The test split is NOT loaded in this step.",
                len(X_train), len(X_val))

    started = perf_counter()
    results = evaluate_candidates(tuned, X_train, y_train, X_val, y_val, methods=methods,
                                  seed=settings.random_seed, n_splits=settings.cv_splits,
                                  n_jobs=args.n_jobs)
    comparison = comparison_table(results)
    business = settings.business
    recall_target = settings.threshold.recall_target
    sweep, choices = threshold_tables(results, y_val, business, recall_target)

    fingerprint = short_fingerprint(settings.raw_data_path)
    out_dir = settings.metrics_dir / "calibration"
    plot_dir = settings.plots_dir / "calibration"
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": settings.random_seed,
        "cv_folds": settings.cv_splits,
        "methods": methods,
        "models": list(tuned),
        "recall_target": recall_target,
        "business_assumptions": {**asdict(business), "status": "PLACEHOLDER"},
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_split_used": False,
        "source_fingerprint": fingerprint,
    }
    files = save_candidates(results, comparison, sweep, choices, y_val, out_dir,
                            settings.candidates_dir, meta=meta)

    if not comparison.empty:
        files.append(plot_reliability(results, y_val, plot_dir / "calibration_curves.png"))
        best = best_calibration_per_model(comparison)
        best_sweeps = {
            f"{REGISTRY[row.model].label if row.model in REGISTRY else row.model} ({row.calibration})":
                sweep[(sweep["model"] == row.model) & (sweep["calibration"] == row.calibration)]
            for row in best.itertuples(index=False)
        }
        files.append(plot_threshold_curves(best_sweeps, business, plot_dir / "threshold_curves.png"))

        if not args.no_learning_curves:
            curves = {}
            for model in best["model"]:
                logger.info("Learning curve: %s", model)
                curves[model] = compute_learning_curve(tuned[model], X_train, y_train,
                                                       seed=settings.random_seed,
                                                       n_splits=settings.cv_splits,
                                                       n_jobs=args.n_jobs)
            lc_path = out_dir / "learning_curves.csv"
            learning_curve_table(curves).to_csv(lc_path, index=False)
            files += [lc_path, plot_learning_curves(curves, plot_dir / "learning_curves.png")]

    report = format_candidates_report(comparison, choices, results, business, recall_target)
    report_path = out_dir / "calibration_report.txt"
    report_path.write_text(report, encoding="utf-8")
    files.append(report_path)
    print("\n" + report)

    if args.no_mlflow:
        logger.info("MLflow logging skipped (--no-mlflow)")
    elif not comparison.empty:
        _log_to_mlflow(results, choices, settings, fingerprint)

    for path in files:
        logger.info("Wrote %s", path)
    logger.info("Finished in %.1f min", (perf_counter() - started) / 60)
    return 0 if any(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())