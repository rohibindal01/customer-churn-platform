"""Milestone 8: customer segmentation.

Fitted on the TRAINING split without the churn label, profiled with it
afterwards, and checked on the validation split. The test split is not used.

Examples (CMD):
    python scripts\\09_run_segmentation.py
    python scripts\\09_run_segmentation.py --k 5
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import joblib
import numpy as np

from churn.analytics.plots import (
    plot_k_selection,
    plot_segment_churn,
    plot_segment_heatmap,
    plot_segments_pca,
)
from churn.analytics.segmentation import (
    BUSINESS_K,
    SEGMENT_FEATURES,
    choose_k,
    compare_methods,
    evaluate_k,
    fit_segment_model,
    name_segments,
    profile_segments,
    structure_label,
)
from churn.config.settings import get_settings
from churn.data.fingerprint import short_fingerprint
from churn.features.splitting import load_split, split_xy
from churn.utils.logger import get_logger

logger = get_logger("run_segmentation")

PROFILE_COLUMNS = ["segment", "name", "customers", "share_pct", "tenure_mean", "monthly_mean",
                   "addons_mean", "month_to_month_pct", "fiber_pct", "no_internet_pct",
                   "no_security_or_support_pct", "churn_rate_pct", "lift", "mean_predicted_pct"]


def _report(choice, k_table, methods, profile_train, stability, overall_pct) -> str:
    cols = [c for c in PROFILE_COLUMNS if c in profile_train.columns]
    lines = [
        "SEGMENTATION REPORT (K-Means on the training split; churn label NOT used to build segments)",
        f"Inputs: {', '.join(SEGMENT_FEATURES)} (standardized).",
        "",
        "1. CHOOSING k",
        k_table.round(4).to_string(index=False),
        "",
        f"   Chosen k = {choice['k']}: {choice['reason']}.",
        f"   Cluster structure: {choice['structure'].upper()} (silhouette {choice['silhouette']:.3f}; "
        "above 0.5 strong, 0.25 to 0.5 reasonable, below 0.25 weak).",
    ]
    if choice["structure"] == "weak":
        lines.append("   WARNING: weak structure. Treat these segments as a convenient partition "
                     "of a continuum, not as naturally separate customer groups.")
    lines += [
        "",
        "2. METHOD COMPARISON (same scaled data; DBSCAN silhouette excludes noise points)",
        methods.round(4).to_string(index=False),
        "",
        f"3. SEGMENT PROFILES (training split; overall churn {overall_pct:.1f}%; "
        "lift = segment churn / overall churn)",
        profile_train[cols].round(2).to_string(index=False),
        "",
        "4. DO THE PROFILES HOLD ON UNSEEN CUSTOMERS? (validation split assigned to the same segments)",
        stability.round(2).to_string(index=False),
        "",
        "CAVEATS",
        "  Segments describe customers; they are not causes of churn.",
        "  A segment's churn rate mixes the effects of all its traits.",
        "  Names are generated from segment averages; individual customers vary.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, help="Force the number of segments (manual override).")
    parser.add_argument("--n-stability", type=int, default=10,
                        help="Bootstrap refits per k for the stability check (default 10).")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    args = parser.parse_args()

    settings = get_settings()
    seed = settings.random_seed
    train = load_split("train", settings=settings)
    val = load_split("validation", settings=settings)
    X_train, y_train = split_xy(train, settings)
    X_val, y_val = split_xy(val, settings)
    logger.info("Segmenting %d training customers; validating on %d. Test split not used.",
                len(X_train), len(X_val))

    # Fit the final model first: its scaler defines the space used everywhere below.
    probe = fit_segment_model(X_train, k=BUSINESS_K[0], seed=seed)
    scaled = probe.transform(X_train)
    k_table = evaluate_k(scaled, seed=seed, n_stability=args.n_stability)
    if args.k:
        row = k_table[k_table["k"] == args.k]
        silhouette = float(row["silhouette"].iloc[0]) if not row.empty else float("nan")
        choice = {"k": args.k, "silhouette": silhouette, "structure": structure_label(silhouette),
                  "reason": "manual override with --k", "constraints_met": None}
    else:
        choice = choose_k(k_table)
    k = choice["k"]
    logger.info("Chosen k = %d (%s structure): %s", k, choice["structure"], choice["reason"])

    model = fit_segment_model(X_train, k=k, seed=seed)
    labels_train = model.predict(X_train)
    methods = compare_methods(scaled, labels_train, k, seed=seed)

    prob_train = prob_val = None
    final_path = settings.models_dir / "final" / "churn_pipeline.joblib"
    if final_path.exists():
        estimator = joblib.load(final_path)
        prob_train = estimator.predict_proba(X_train)[:, 1]
        prob_val = estimator.predict_proba(X_val)[:, 1]
    else:
        logger.warning("No final model found; profiles will not include predicted risk.")

    profile_train = profile_segments(X_train, labels_train, y_train, prob_train)
    model.names = name_segments(profile_train, model.spend_cuts)
    profile_train.insert(1, "name", profile_train["segment"].map(model.names))

    labels_val = model.predict(X_val)
    profile_val = profile_segments(X_val, labels_val, y_val, prob_val)
    profile_val.insert(1, "name", profile_val["segment"].map(model.names))
    stability = profile_train[["segment", "name", "customers", "churn_rate_pct"]].merge(
        profile_val[["segment", "customers", "churn_rate_pct"]], on="segment", how="left",
        suffixes=("_train", "_validation"))
    stability["churn_gap_pts"] = (stability["churn_rate_pct_validation"]
                                  - stability["churn_rate_pct_train"])

    out_dir = settings.segmentation_dir
    plot_dir = settings.plots_dir / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "segment_model.joblib"
    joblib.dump(model, model_path)
    files = [model_path]
    for name, table in {"k_selection": k_table, "method_comparison": methods,
                        "segment_profile_train": profile_train,
                        "segment_profile_validation": profile_val,
                        "segment_stability": stability}.items():
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        files.append(path)

    overall_pct = float(np.mean(y_train) * 100)
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "k": k, "choice": choice, "features": SEGMENT_FEATURES,
        "names": {str(s): n for s, n in model.names.items()},
        "spend_cuts": list(model.spend_cuts), "fitted_on": "train split",
        "churn_label_used_for_fitting": False, "random_seed": seed,
        "source_fingerprint": short_fingerprint(settings.raw_data_path),
    }
    meta_path = out_dir / "segmentation_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    files.append(meta_path)

    files += [
        plot_k_selection(k_table, k, plot_dir / "k_selection.png"),
        plot_segments_pca(scaled, labels_train, model.names, plot_dir / "segments_pca.png", seed=seed),
        plot_segment_churn(profile_train, overall_pct, plot_dir / "segment_churn.png"),
        plot_segment_heatmap(profile_train, plot_dir / "segment_profiles.png"),
    ]
    report = _report(choice, k_table, methods, profile_train, stability, overall_pct)
    report_path = out_dir / "segmentation_report.txt"
    report_path.write_text(report, encoding="utf-8")
    files.append(report_path)
    print("\n" + report)

    if not args.no_mlflow:
        try:
            from churn.utils.tracking import log_run

            row = k_table[k_table["k"] == k].iloc[0]
            log_run(tracking_uri=settings.mlflow_tracking_uri,
                    experiment_name=settings.mlflow.experiment_name,
                    artifact_dir=settings.root / "mlartifacts", run_name=f"segmentation_k{k}",
                    params={"k": k, "features": ",".join(SEGMENT_FEATURES),
                            "selected_by": choice["reason"][:200]},
                    metrics={"silhouette": float(row["silhouette"]),
                             "stability_ari": float(row["stability_ari"]),
                             "davies_bouldin": float(row["davies_bouldin"])},
                    tags={"stage": "segmentation", "split": "train",
                          "data_fingerprint": meta["source_fingerprint"]},
                    artifacts=[report_path])
        except Exception as exc:  # noqa: BLE001 - tracking must never break the analysis
            logger.warning("MLflow logging failed: %s", exc)

    for path in files:
        logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())