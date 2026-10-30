"""Milestone 6B: select the final model on VALIDATION data, then (only with
--evaluate-test) evaluate it ONCE on the test split and save it.

Examples (CMD):
    python scripts\\07_finalize_model.py
    python scripts\\07_finalize_model.py --evaluate-test
    python scripts\\07_finalize_model.py --model gradient_boosting --calibration uncalibrated
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from churn.config.settings import get_settings
from churn.data import schema
from churn.data.fingerprint import short_fingerprint
from churn.features.preprocessing import get_feature_names
from churn.features.splitting import load_split, split_xy
from churn.models.final import (
    FINAL_MODEL_FILE,
    base_pipeline,
    check_not_finalized,
    evaluate_on_test,
    format_final_report,
    package_versions,
    plot_test_calibration,
    plot_test_confusion,
    plot_test_curves,
    save_final,
)
from churn.models.registry import REGISTRY
from churn.models.selection import (
    SELECTION_RULE,
    candidate_variants,
    format_selection_report,
    select_final,
    selection_table,
)
from churn.models.threshold import POLICIES
from churn.utils.logger import get_logger

logger = get_logger("finalize_model")


def _log_to_mlflow(settings, metadata, evaluation, model_path) -> None:
    from churn.utils.tracking import log_run

    try:
        ci = evaluation["confidence_intervals"]
        chosen = evaluation["policies"][evaluation["chosen_policy"]]
        metrics = {f"test_{k}": ci[k]["value"] for k in ci}
        metrics.update({f"test_{k}_ci_low": ci[k]["ci_low"] for k in ci})
        metrics.update({f"test_{k}_ci_high": ci[k]["ci_high"] for k in ci})
        for key in ("precision", "recall", "f1", "specificity", "balanced_accuracy", "mcc",
                    "accuracy", "flagged_pct", "net_benefit_per_1000"):
            metrics[f"test_{key}_at_threshold"] = chosen[key]
        run_id = log_run(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow.experiment_name,
            artifact_dir=settings.root / "mlartifacts",
            run_name=f"final_{metadata['model']}__{metadata['calibration']}",
            params={
                "model": metadata["model"],
                "calibration": metadata["calibration"],
                "threshold_policy": metadata["threshold"]["policy"],
                "threshold": metadata["threshold"]["value"],
                **metadata["hyperparameters"],
            },
            metrics=metrics,
            tags={
                "stage": "final",
                "split": "test",
                "selected_by": metadata["selected_by"],
                "data_fingerprint": metadata["data"]["source_fingerprint"],
                "test_evaluation_count": metadata["test_evaluation_count"],
                "business_assumptions": "placeholder",
                "model_artifact": str(model_path),
            },
        )
        logger.info("MLflow run %s logged for the final model", run_id)
    except Exception as exc:  # noqa: BLE001 - tracking must never break finalization
        logger.warning("MLflow logging failed: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Open the test split ONCE, evaluate and save the final model.")
    parser.add_argument("--model", help="Manual override: model key (needs --calibration).")
    parser.add_argument("--calibration", help="Manual override: uncalibrated, sigmoid or isotonic.")
    parser.add_argument("--policy", default="max_net_benefit", choices=POLICIES,
                        help="Threshold policy stored as the default (default: max_net_benefit).")
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples (default 2000).")
    parser.add_argument("--force", action="store_true",
                        help="Allow re-evaluating on the test split. The rerun is recorded.")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    args = parser.parse_args()

    if (args.model is None) != (args.calibration is None):
        logger.error("Give both --model and --calibration, or neither.")
        return 1

    settings = get_settings()
    calib_dir = settings.metrics_dir / "calibration"
    paths = {name: calib_dir / name for name in
             ("validation_comparison.csv", "threshold_choices.csv", "validation_probabilities.npz")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        logger.error("Missing Milestone 6A outputs %s. Run scripts\\06_calibrate_and_threshold.py",
                     missing)
        return 1

    comparison = pd.read_csv(paths["validation_comparison.csv"])
    choices = pd.read_csv(paths["threshold_choices.csv"])
    npz = np.load(paths["validation_probabilities.npz"])
    y_val = npz["y_val"].astype(int)
    val_probs = {k: npz[k] for k in npz.files if k != "y_val"}

    # ---- Selection on validation data only ------------------------------
    variants = candidate_variants(comparison)
    table = selection_table(variants, val_probs, y_val, n_boot=args.n_boot,
                            seed=settings.random_seed)
    decision = select_final(table, args.model, args.calibration)
    model, calibration = decision["model"], decision["calibration"]
    key = f"{model}__{calibration}"

    candidate_path = settings.candidates_dir / f"{key}.joblib"
    policy_rows = choices[(choices["model"] == model)
                          & (choices["calibration"] == calibration)].reset_index(drop=True)
    if not candidate_path.exists() or policy_rows.empty:
        logger.error("No saved candidate or thresholds for %s. Check the names, or rerun Milestone 6A.",
                     key)
        return 1
    thresholds = {row.policy: float(row.threshold) for row in policy_rows.itertuples(index=False)}
    threshold = thresholds[args.policy]

    out_dir = settings.metrics_dir / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_text = format_selection_report(table, decision, policy_rows, args.policy,
                                             settings.business)
    table.to_csv(out_dir / "selection_evidence.csv", index=False)
    (out_dir / "selection_report.txt").write_text(selection_text, encoding="utf-8")
    print("\n" + selection_text)

    if not args.evaluate_test:
        logger.info("Dry run complete: the test split was NOT loaded.")
        logger.info("Review the decision above, then run: python scripts\\07_finalize_model.py --evaluate-test")
        return 0

    # ---- One-time test evaluation ---------------------------------------
    final_dir = settings.models_dir / "final"
    try:
        previous_count = check_not_finalized(final_dir, args.force)
    except FileExistsError as exc:
        logger.error("%s", exc)
        return 1

    estimator = joblib.load(candidate_path)
    X_test, y_test = split_xy(load_split("test", settings=settings), settings)
    logger.info("Evaluating %s on the test split (%d rows). Evaluation number %d.",
                key, len(X_test), previous_count + 1)
    evaluation = evaluate_on_test(estimator, X_test, y_test, thresholds, args.policy,
                                  settings.business, n_boot=args.n_boot, seed=settings.random_seed)

    params_path = settings.metrics_dir / "tuning" / "best_params.json"
    best_params = (json.loads(params_path.read_text(encoding="utf-8")).get(model, {})
                   if params_path.exists() else {})
    val_row = comparison[(comparison["model"] == model)
                         & (comparison["calibration"] == calibration)].iloc[0]
    val_policy = policy_rows.set_index("policy").loc[args.policy]
    manifest_path = settings.processed_dir / "split_manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {})

    metadata = {
        "model_name": "churn_pipeline",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "model_label": REGISTRY[model].label if model in REGISTRY else model,
        "calibration": calibration,
        "imbalance_strategy": "none",
        "selected_by": decision["selected_by"],
        "selection_reason": decision["reason"],
        "selection_rule": list(SELECTION_RULE),
        "hyperparameters": best_params,
        "threshold": {"policy": args.policy, "value": threshold,
                      "all_policies": thresholds, "chosen_on": "validation split"},
        "business_assumptions": {**asdict(settings.business), "status": "PLACEHOLDER"},
        "input_columns": list(schema.RAW_FEATURE_COLUMNS),
        "feature_names": get_feature_names(base_pipeline(estimator)),
        "validation_metrics": {
            **{k: float(val_row[k]) for k in ("roc_auc", "pr_auc", "brier", "log_loss", "ece")},
            "at_threshold": {k: float(val_policy[k]) for k in
                             ("threshold", "precision", "recall", "f1", "flagged_pct",
                              "net_benefit_per_1000")},
        },
        "test_metrics": {
            "probability": evaluation["probability_metrics"],
            "confidence_intervals": evaluation["confidence_intervals"],
            "at_threshold": evaluation["policies"][args.policy],
        },
        "data": {
            "source_fingerprint": short_fingerprint(settings.raw_data_path),
            "random_seed": settings.random_seed,
            "splits": manifest.get("splits", {}),
        },
        "training": {"fitted_on": "train split only",
                     "artifact": f"artifacts/models/final/{FINAL_MODEL_FILE}"},
        "test_evaluation_count": previous_count + 1,
        "forced_rerun": bool(args.force and previous_count > 0),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "versions": package_versions()},
    }

    files = save_final(estimator, metadata, evaluation, final_dir, out_dir)
    plot_dir = settings.plots_dir / "final"
    y_true, prob = evaluation["y_true"], evaluation["probabilities"]
    files += [
        plot_test_curves(y_true, prob, threshold, plot_dir / "test_roc_pr.png"),
        plot_test_calibration(y_true, prob, plot_dir / "test_calibration.png"),
        plot_test_confusion(y_true, prob, threshold, plot_dir / "test_confusion_matrix.png"),
    ]
    report = format_final_report(metadata, evaluation)
    report_path = out_dir / "final_report.txt"
    report_path.write_text(report, encoding="utf-8")
    files.append(report_path)
    print("\n" + report)

    if not args.no_mlflow:
        _log_to_mlflow(settings, metadata, evaluation, final_dir / FINAL_MODEL_FILE)
    for path in files:
        logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())