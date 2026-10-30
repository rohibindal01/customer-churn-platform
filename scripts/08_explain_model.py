"""Milestone 7: explain the final churn model.

The training split is the SHAP background; every explanation uses the
VALIDATION split. The test split is not used.

Examples (CMD):
    python scripts\\08_explain_model.py
    python scripts\\08_explain_model.py --customer-id 1234-ABCDE --n-jobs 8
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata

import joblib
import numpy as np

from churn.config.settings import get_settings
from churn.data.fingerprint import short_fingerprint
from churn.explain.permutation import compute_permutation_importance
from churn.explain.plots import (
    plot_bar,
    plot_local_explanation,
    plot_shap_summary,
    plot_signed_bar,
)
from churn.explain.shap_explainer import (
    ChurnExplainer,
    explain_customer,
    global_feature_table,
    global_group_table,
    model_specific_importance,
    pick_example_customers,
)
from churn.features.splitting import load_split, split_xy
from churn.models.final import FINAL_MODEL_FILE, METADATA_FILE
from churn.utils.logger import get_logger

logger = get_logger("explain_model")

CAVEATS = (
    "Importance shows what the MODEL relies on, not what CAUSES churn. Changing a feature for a",
    "  customer is not proven to change their behaviour without an intervention test.",
    "Correlated features (tenure, TotalCharges, avg_monthly_spend, tenure_group) share and split",
    "  importance; each can look smaller than the information they carry together.",
    "Shuffling a raw column in permutation importance also changes the features engineered from it.",
    "SHAP contributions are relative to the average customer in the training background sample.",
)


def _parse_list(value: str | None) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()] if value else []


def _report(meta, perm, group_table, feature_table, model_imp, local_results, n_repeats) -> str:
    units = meta["units"]
    lines = [
        f"EXPLAINABILITY REPORT: {meta['model_label']} ({meta['calibration']}), "
        f"threshold {meta['threshold']:.2f}",
        f"Data: validation split ({meta['explained_rows']} customers explained). "
        f"SHAP background: {meta['background_rows']} training customers.",
        f"Explainer: {meta['explainer']} ({units}). Additivity check, max |baseline + sum(SHAP) "
        f"- model output|: {meta['additivity_max_error']:.2e}",
        "The test split is not used here.",
        "",
        f"1. PERMUTATION IMPORTANCE (drop in validation PR-AUC when one raw input column is "
        f"shuffled, {n_repeats} repeats)",
        perm[["feature", "importance_mean", "importance_std", "share_positive"]]
        .head(12).round(4).to_string(index=False),
        "   share_positive = fraction of repeats in which shuffling made the model worse.",
        "",
        "2. SHAP IMPORTANCE BY MODEL FEATURE (mean |SHAP|; one-hot columns summed per feature)",
        group_table.head(15).round(4).to_string(index=False),
        "",
        "3. DIRECTION OF EFFECT (top 15 preprocessed features; mean SHAP when high/present vs "
        "low/absent)",
        feature_table[["feature", "mean_abs_shap", "effect_high", "effect_low", "direction"]]
        .head(15).round(4).to_string(index=False),
        "",
    ]
    if not model_imp.empty and "coefficient" in model_imp.columns:
        non_zero = int((~model_imp["is_zero"]).sum())
        lines += [
            f"4. MODEL COEFFICIENTS ({non_zero} of {len(model_imp)} non-zero). Numeric features are "
            "standardized: a coefficient is the change in log-odds per 1 standard deviation.",
            model_imp[["feature", "coefficient", "odds_ratio"]].head(12).round(4).to_string(index=False),
            "",
        ]
    elif not model_imp.empty:
        lines += [
            "4. MODEL-SPECIFIC IMPORTANCE (impurity-based; favours features with many split points)",
            model_imp[["feature", "importance"]].head(12).round(4).to_string(index=False),
            "",
        ]
    lines.append("5. INDIVIDUAL CUSTOMERS (validation split)")
    for item in local_results:
        status = "FLAGGED" if item["flagged"] else "not flagged"
        lines += [
            f"  [{item['label']}] customer {item['customer_id']}: churn probability "
            f"{item['probability']:.3f} -> {status} at threshold {meta['threshold']:.2f}",
            f"    average customer {item['base_value']:+.3f} {units}; "
            f"this customer {item['model_output']:+.3f} {units}",
        ]
        for r in item["contributions"].itertuples(index=False):
            value = f" = {r.value}" if r.value != "" else ""
            odds = f"   (odds x{r.odds_multiplier:.2f})" if hasattr(r, "odds_multiplier") else ""
            lines.append(f"    {r.shap:+.3f}  {r.group}{value}{odds}")
        lines.append("")
    lines += ["CAVEATS", *[f"  {c}" for c in CAVEATS]]
    return "\n".join(lines)


def _log_to_mlflow(settings, meta, perm, report_path) -> None:
    from churn.utils.tracking import log_run

    try:
        metrics = {f"perm_{row.feature}": row.importance_mean
                   for row in perm.head(5).itertuples(index=False)}
        metrics["additivity_max_error"] = meta["additivity_max_error"]
        run_id = log_run(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow.experiment_name,
            artifact_dir=settings.root / "mlartifacts",
            run_name=f"explain_{meta['model']}",
            params={"explainer": meta["explainer"], "background_rows": meta["background_rows"],
                    "explained_rows": meta["explained_rows"], "n_repeats": meta["n_repeats"]},
            metrics=metrics,
            tags={"stage": "explain", "model": meta["model"], "split": "validation",
                  "data_fingerprint": meta["source_fingerprint"]},
            artifacts=[report_path],
        )
        logger.info("MLflow run %s logged", run_id)
    except Exception as exc:  # noqa: BLE001 - tracking must never break the analysis
        logger.warning("MLflow logging failed: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-repeats", type=int, default=20, help="Permutation repeats (default 20).")
    parser.add_argument("--background", type=int, default=200,
                        help="Training customers used as the SHAP background (default 200).")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Explain at most this many validation rows (0 = all).")
    parser.add_argument("--customer-id", help="Comma-separated validation customerIDs to explain too.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for permutation importance.")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    args = parser.parse_args()

    settings = get_settings()
    final_dir = settings.models_dir / "final"
    model_path, meta_path = final_dir / FINAL_MODEL_FILE, final_dir / METADATA_FILE
    if not model_path.exists() or not meta_path.exists():
        logger.error("No final model in %s. Run: python scripts\\07_finalize_model.py --evaluate-test",
                     final_dir)
        return 1
    estimator = joblib.load(model_path)
    final_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    threshold = float(final_meta["threshold"]["value"])

    X_train, _ = split_xy(load_split("train", settings=settings), settings)
    val = load_split("validation", settings=settings)
    X_val, y_val = split_xy(val, settings)
    ids_val = val[settings.data.id_column].astype(str).to_numpy()
    logger.info("Explaining %s on the validation split (%d rows). The test split is not used.",
                final_meta["model"], len(X_val))

    # 1. Permutation importance (model-agnostic, raw input columns)
    perm = compute_permutation_importance(estimator, X_val, y_val, n_repeats=args.n_repeats,
                                          seed=settings.random_seed, n_jobs=args.n_jobs)

    # 2. SHAP
    explainer = ChurnExplainer(estimator, X_train, max_background=args.background,
                               seed=settings.random_seed)
    if args.max_rows and len(X_val) > args.max_rows:
        rng = np.random.default_rng(settings.random_seed)
        positions = np.sort(rng.choice(len(X_val), size=args.max_rows, replace=False))
    else:
        positions = np.arange(len(X_val))
    X_rows = X_val.iloc[positions]
    shap_values = explainer.shap_values(X_rows)
    X_t = explainer.transform(X_rows)
    additivity = float(np.max(np.abs(explainer.expected_value + shap_values.sum(axis=1)
                                     - explainer.model_output(X_rows))))
    logger.info("SHAP (%s) on %d rows; additivity max error %.2e", explainer.kind, len(X_rows),
                additivity)

    feature_table = global_feature_table(shap_values, X_t, explainer.feature_names, explainer.groups)
    group_table = global_group_table(shap_values, explainer.groups)
    model_imp = model_specific_importance(explainer)

    # 3. Individual customers
    prob_val = np.asarray(estimator.predict_proba(X_val))[:, 1]
    examples = pick_example_customers(prob_val, threshold)
    for customer_id in _parse_list(args.customer_id):
        found = np.where(ids_val == customer_id)[0]
        if found.size:
            examples[f"requested_{customer_id}"] = int(found[0])
        else:
            logger.warning("customerID %s is not in the validation split; skipped", customer_id)

    local_results, local_rows = [], []
    for label, position in examples.items():
        result = explain_customer(explainer, X_val.iloc[[position]])
        item = {"label": label, "customer_id": ids_val[position],
                "probability": float(prob_val[position]),
                "flagged": bool(prob_val[position] >= threshold), **result}
        local_results.append(item)
        table = result["contributions"].copy()
        table.insert(0, "probability", item["probability"])
        table.insert(0, "customer_id", item["customer_id"])
        table.insert(0, "example", label)
        local_rows.append(table)

    # 4. Save tables
    shap_dir = settings.shap_dir
    plot_dir = settings.plots_dir / "explain"
    shap_dir.mkdir(parents=True, exist_ok=True)
    files = []
    tables = {"permutation_importance": perm, "shap_global_features": feature_table,
              "shap_global_groups": group_table, "model_importance": model_imp}
    for name, table in tables.items():
        path = shap_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        files.append(path)
    local_path = shap_dir / "local_explanations.csv"
    (local_rows[0].iloc[0:0] if not local_rows else
     __import__("pandas").concat(local_rows, ignore_index=True)).to_csv(local_path, index=False)
    files.append(local_path)

    npz_path = shap_dir / "shap_values_validation.npz"
    np.savez_compressed(
        npz_path, shap_values=shap_values, X_transformed=X_t,
        feature_names=np.asarray(explainer.feature_names, dtype=str),
        groups=np.asarray(explainer.groups, dtype=str),
        customer_ids=ids_val[positions].astype(str),
        probability=prob_val[positions], base_value=np.asarray([explainer.expected_value]),
    )
    files.append(npz_path)

    try:
        shap_version = importlib_metadata.version("shap")
    except importlib_metadata.PackageNotFoundError:
        shap_version = "unknown"
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": final_meta["model"], "model_label": final_meta["model_label"],
        "calibration": final_meta["calibration"], "threshold": threshold,
        "explainer": explainer.kind, "units": explainer.units,
        "background_rows": explainer.background_rows, "explained_rows": int(len(X_rows)),
        "additivity_max_error": additivity, "n_repeats": args.n_repeats,
        "permutation_scoring": "average_precision", "split": "validation",
        "examples": {label: str(ids_val[pos]) for label, pos in examples.items()},
        "source_fingerprint": short_fingerprint(settings.raw_data_path),
        "shap_version": shap_version,
    }
    meta_out = shap_dir / "explain_meta.json"
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    files.append(meta_out)

    # 5. Plots
    top_perm = perm.head(15)
    files.append(plot_bar(top_perm["feature"], top_perm["importance_mean"],
                          plot_dir / "permutation_importance.png",
                          "Permutation importance (validation split)",
                          "Drop in PR-AUC when the column is shuffled",
                          errors=top_perm["importance_std"]))
    top_groups = group_table.head(15)
    files.append(plot_bar(top_groups["group"], top_groups["mean_abs_shap"],
                          plot_dir / "shap_importance_groups.png",
                          "SHAP importance per model feature (one-hot columns summed)",
                          f"Mean |SHAP| ({explainer.units})"))
    files.append(plot_shap_summary(shap_values, X_t, explainer.feature_names,
                                   plot_dir / "shap_summary.png", units=explainer.units))
    if not model_imp.empty and "coefficient" in model_imp.columns:
        top_coef = model_imp[~model_imp["is_zero"]].head(20)
        files.append(plot_signed_bar(top_coef["feature"], top_coef["coefficient"],
                                     plot_dir / "coefficients.png",
                                     "Logistic regression coefficients (largest 20)",
                                     "Change in log-odds (numeric: per 1 standard deviation)"))
    elif not model_imp.empty:
        top_imp = model_imp.head(20)
        files.append(plot_bar(top_imp["feature"], top_imp["importance"],
                              plot_dir / "model_importance.png",
                              "Model-specific importance", "Importance"))
    for item in local_results:
        files.append(plot_local_explanation(
            item["contributions"], plot_dir / f"local_{item['label']}.png",
            f"{item['label']}: customer {item['customer_id']}, "
            f"churn probability {item['probability']:.2f}", units=explainer.units))

    report = _report(meta, perm, group_table, feature_table, model_imp, local_results,
                     args.n_repeats)
    report_path = shap_dir / "explain_report.txt"
    report_path.write_text(report, encoding="utf-8")
    files.append(report_path)
    print("\n" + report)

    if not args.no_mlflow:
        _log_to_mlflow(settings, meta, perm, report_path)
    for path in files:
        logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())