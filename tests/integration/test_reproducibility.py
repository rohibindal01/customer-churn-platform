"""Reproducibility and end-to-end checks on the real dataset."""
import json

import numpy as np
import pytest

from churn.config.settings import get_settings
from churn.data.fingerprint import short_fingerprint
from churn.data.loader import load_raw_data, load_validated_data
from churn.features.cleaning import drop_duplicate_rows
from churn.features.splitting import load_split, stratified_split
from churn.serving.predictor import PredictionService

SETTINGS = get_settings()
MANIFEST = SETTINGS.processed_dir / "split_manifest.json"

pytestmark = pytest.mark.skipif(
    not (SETTINGS.raw_data_path.exists() and MANIFEST.exists()),
    reason="Raw data or saved splits missing (run scripts 01 and 03)",
)


def test_data_fingerprint_matches_the_recorded_one():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert short_fingerprint(SETTINGS.raw_data_path) == manifest["source_fingerprint"]


def test_split_is_exactly_reproducible():
    df, _ = drop_duplicate_rows(load_raw_data())
    for name, frame in zip(("train", "validation", "test"), stratified_split(df)):
        saved = load_split(name)
        assert frame["customerID"].astype(str).tolist() == saved["customerID"].astype(str).tolist()


@pytest.mark.skipif(not (SETTINGS.models_dir / "final" / "churn_pipeline.joblib").exists(),
                    reason="Final model missing")
def test_raw_rows_flow_end_to_end():
    raw = load_validated_data().sample(n=50, random_state=0)
    service = PredictionService.from_artifacts()
    out = service.predict_batch(raw.drop(columns=[SETTINGS.data.target_column]))
    assert (out["status"] == "scored").all()
    assert out["churn_probability"].between(0, 1).all()
    assert set(out["risk_band"]) <= {"Low", "Medium", "High", "Critical"}
    if service.segment_model is not None:
        assert (out["segment"] != "").all()
    assert np.isfinite(out["churn_probability"].to_numpy(dtype=float)).all()