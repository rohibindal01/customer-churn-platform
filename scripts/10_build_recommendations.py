"""Milestone 8: risk bands and rule-based retention recommendations.

Applied to the VALIDATION split, so each band's observed churn can be
checked. The test split is not used.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from churn.analytics.plots import plot_risk_bands, plot_rule_frequency
from churn.analytics.recommendations import (
    DISCLAIMER,
    RULES,
    RuleConfig,
    customer_records,
    family_contributions,
    recommend,
)
from churn.analytics.risk import RISK_BANDS, assign_risk_band, band_summary, bands_from_metadata
from churn.config.settings import get_settings
from churn.explain.shap_explainer import ChurnExplainer
from churn.features.cleaning import clean_features
from churn.features.splitting import load_split, split_xy
from churn.models.final import FINAL_MODEL_FILE, METADATA_FILE
from churn.utils.logger import get_logger

logger = get_logger("build_recommendations")


def _report(thresholds, config, summary, rule_table, crosstab, examples) -> str:
    lines = [
        "RETENTION RECOMMENDATIONS (validation split; the test split is not used)",
        "",
        "1. RISK BANDS (thresholds from the final model's metadata)",
        f"   Low < {thresholds.watch:.2f} <= Medium < {thresholds.decision:.2f} <= High < "
        f"{thresholds.critical:.2f} <= Critical",
        summary.round(1).to_string(index=False),
        "   Observed churn should rise from band to band and sit near the mean predicted value.",
        "",
        f"2. RULES TRIGGERED (customers in Medium, High or Critical bands; high-spend cut-off = "
        f"{config.high_spend_monthly:.2f}, the training 75th percentile)",
        rule_table.to_string(index=False),
        "",
        "3. RISK BAND MIX PER SEGMENT (% of each segment's customers)",
        crosstab.round(1).to_string(),
        "",
        "4. EXAMPLE CUSTOMERS (three highest predicted risks)",
    ]
    for ex in examples:
        lines.append(f"  customer {ex['customer_id']} | {ex['band']} risk, probability "
                     f"{ex['probability']:.3f} | {ex['segment']}")
        for a in ex["actions"]:
            driver = (f"{a['driver_family']} {a['driver_contribution']:+.2f} log-odds for this customer"
                      if a["driver_contribution"] is not None else a["driver_family"])
            lines += [f"    - {a['title']} [{a['basis']}; {driver}]",
                      f"      Action: {a['action']}",
                      f"      Why: {a['rationale']}"]
        lines.append("")
    lines += ["DISCLAIMER", f"  {DISCLAIMER}"]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=3, help="Suggestions kept per customer (default 3).")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    args = parser.parse_args()

    settings = get_settings()
    final_dir = settings.models_dir / "final"
    segment_path = settings.segmentation_dir / "segment_model.joblib"
    for path in (final_dir / FINAL_MODEL_FILE, final_dir / METADATA_FILE, segment_path):
        if not path.exists():
            logger.error("Missing %s. Run Milestones 6B (07) and 8 (09) first.", path)
            return 1
    estimator = joblib.load(final_dir / FINAL_MODEL_FILE)
    metadata = json.loads((final_dir / METADATA_FILE).read_text(encoding="utf-8"))
    segment_model = joblib.load(segment_path)

    X_train, _ = split_xy(load_split("train", settings=settings), settings)
    val = load_split("validation", settings=settings)
    X_val, y_val = split_xy(val, settings)
    ids = val[settings.data.id_column].astype(str).to_numpy()

    thresholds = bands_from_metadata(metadata)
    high_spend = float(np.quantile(clean_features(X_train)["MonthlyCharges"], 0.75))
    config = RuleConfig(high_spend_monthly=high_spend)

    prob = np.asarray(estimator.predict_proba(X_val))[:, 1]
    bands = assign_risk_band(prob, thresholds)
    explainer = ChurnExplainer(estimator, X_train, seed=settings.random_seed)
    families = family_contributions(explainer.shap_values(X_val), explainer.groups)
    records = customer_records(X_val)
    segments = segment_model.segment_names(X_val)

    rows, matches, results = [], [], []
    for i in range(len(X_val)):
        contrib = families.iloc[i].to_dict()
        rec = recommend(records[i], prob[i], str(bands[i]), contrib, config, max_items=args.top)
        top_family = max(contrib, key=contrib.get)
        row = {"customer_id": ids[i], "probability": float(prob[i]), "risk_band": str(bands[i]),
               "segment": segments[i],
               "top_driver": top_family if contrib[top_family] > 0 else "",
               "top_driver_contribution": float(contrib[top_family]),
               "rules_matched": len(rec["matched_rule_ids"]), "summary": rec["summary"]}
        for j in range(args.top):
            row[f"action_{j + 1}"] = rec["actions"][j]["title"] if j < len(rec["actions"]) else ""
        rows.append(row)
        matches += [{"band": str(bands[i]), "rule_id": r} for r in rec["matched_rule_ids"]]
        results.append({**rec, "customer_id": ids[i], "segment": segments[i]})

    table = pd.DataFrame(rows)
    summary = band_summary(prob, bands, y_val)
    titles = {r.rule_id: r.title for r in RULES}
    match_frame = pd.DataFrame(matches, columns=["band", "rule_id"])
    rule_table = (match_frame.assign(title=match_frame["rule_id"].map(titles))
                  .pivot_table(index="title", columns="band", values="rule_id",
                               aggfunc="count", fill_value=0)
                  .reindex(columns=[b for b in RISK_BANDS if b in set(match_frame["band"])],
                           fill_value=0))
    rule_table["total"] = rule_table.sum(axis=1)
    rule_table = rule_table.sort_values("total", ascending=False).reset_index()
    crosstab = pd.crosstab(table["segment"], table["risk_band"], normalize="index") * 100
    crosstab = crosstab.reindex(columns=[b for b in RISK_BANDS if b in crosstab.columns])
    order = np.argsort(prob)[::-1][:3]
    examples = [results[i] for i in order]

    out_dir = settings.artifacts_dir / "recommendations"
    plot_dir = settings.plots_dir / "recommendations"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name, frame in {"customer_recommendations_validation": table,
                        "risk_band_summary": summary, "rule_summary": rule_table,
                        "segment_risk_mix": crosstab.reset_index()}.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        files.append(path)

    config_out = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "band_thresholds": asdict(thresholds), "rule_config": asdict(config),
        "rules": [{"rule_id": r.rule_id, "title": r.title, "family": r.family, "basis": r.basis,
                   "priority": r.priority, "bands": list(r.bands), "action": r.action,
                   "rationale": r.rationale} for r in RULES],
        "disclaimer": DISCLAIMER,
    }
    config_path = out_dir / "recommendation_config.json"
    config_path.write_text(json.dumps(config_out, indent=2), encoding="utf-8")
    files.append(config_path)

    files.append(plot_risk_bands(summary, plot_dir / "risk_bands.png"))
    if not rule_table.empty:
        files.append(plot_rule_frequency(rule_table.set_index("title")["total"],
                                         plot_dir / "rule_frequency.png"))
    report = _report(thresholds, config, summary, rule_table, crosstab, examples)
    report_path = out_dir / "recommendations_report.txt"
    report_path.write_text(report, encoding="utf-8")
    files.append(report_path)
    print("\n" + report)

    if not args.no_mlflow:
        try:
            from churn.utils.tracking import log_run

            metrics = {f"share_{r.band.lower()}_pct": r.share_pct
                       for r in summary.itertuples(index=False)}
            metrics.update({f"observed_churn_{r.band.lower()}_pct": r.observed_churn_pct
                            for r in summary.itertuples(index=False)})
            log_run(tracking_uri=settings.mlflow_tracking_uri,
                    experiment_name=settings.mlflow.experiment_name,
                    artifact_dir=settings.root / "mlartifacts", run_name="recommendations",
                    params={"watch": thresholds.watch, "decision": thresholds.decision,
                            "critical": thresholds.critical, "high_spend": round(high_spend, 2)},
                    metrics=metrics,
                    tags={"stage": "recommendations", "split": "validation"},
                    artifacts=[report_path])
        except Exception as exc:  # noqa: BLE001 - tracking must never break the analysis
            logger.warning("MLflow logging failed: %s", exc)

    for path in files:
        logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())