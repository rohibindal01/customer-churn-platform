"""Milestone 9: build the small serving artifacts used by the app and Docker.

Writes to artifacts\\serving:
  background.csv                 200 training customers (SHAP background, raw inputs only)
  portfolio_validation.csv       held-out validation customers scored by PredictionService
  sample_batch.csv               25 validation customers without the Churn column
  sample_batch_with_errors.csv   5 rows, 3 deliberately invalid (demo of row-level checks)
  serving_manifest.json
The test split is not used.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pandas as pd

from churn.config.settings import get_settings
from churn.data.fingerprint import short_fingerprint
from churn.features.splitting import load_split, split_xy
from churn.models.final import FINAL_MODEL_FILE
from churn.serving.predictor import BACKGROUND_FILE, PredictionService
from churn.utils.logger import get_logger

logger = get_logger("prepare_serving")


def main() -> int:
    settings = get_settings()
    if not (settings.models_dir / "final" / FINAL_MODEL_FILE).exists():
        logger.error("No final model. Run: python scripts\\07_finalize_model.py --evaluate-test")
        return 1
    out_dir = settings.artifacts_dir / "serving"
    out_dir.mkdir(parents=True, exist_ok=True)
    target, id_col = settings.data.target_column, settings.data.id_column

    train = load_split("train", settings=settings)
    val = load_split("validation", settings=settings)
    X_train, _ = split_xy(train, settings)

    background = X_train.sample(n=min(200, len(X_train)), random_state=settings.random_seed)
    background_path = out_dir / BACKGROUND_FILE
    background.to_csv(background_path, index=False)

    service = PredictionService.from_artifacts(settings)
    features = val.drop(columns=[target])
    scored = service.predict_batch(features, explain=True)
    rejected = int((scored["status"] != "scored").sum())
    if rejected:
        logger.warning("%d validation rows were rejected; check the data.", rejected)
    extra = scored.drop(columns=["row", "status", "problem", id_col], errors="ignore")
    portfolio = pd.concat([val.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    portfolio_path = out_dir / "portfolio_validation.csv"
    portfolio.to_csv(portfolio_path, index=False)

    sample_path = out_dir / "sample_batch.csv"
    features.head(25).to_csv(sample_path, index=False)

    broken = features.head(5).astype(object).reset_index(drop=True)
    broken.loc[0, "Contract"] = "Weekly"
    broken.loc[1, "tenure"] = "-3"
    broken.loc[2, "MonthlyCharges"] = "abc"
    errors_path = out_dir / "sample_batch_with_errors.csv"
    broken.to_csv(errors_path, index=False)
    check = service.predict_batch(pd.read_csv(errors_path, dtype=str, keep_default_na=False),
                                  explain=False)
    logger.info("Sample with errors: %d of %d rows rejected (expected 3)",
                int((check["status"] == "rejected").sum()), len(check))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": service.metadata.get("model"),
        "threshold": service.threshold,
        "background_rows": int(len(background)),
        "portfolio_rows": int(len(portfolio)),
        "portfolio_rejected_rows": rejected,
        "portfolio_split": "validation",
        "test_split_used": False,
        "source_fingerprint": short_fingerprint(settings.raw_data_path),
    }
    manifest_path = out_dir / "serving_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for path in (background_path, portfolio_path, sample_path, errors_path, manifest_path):
        logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())