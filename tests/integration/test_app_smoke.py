"""Smoke tests on the real artifacts: the service scores the samples and every
app page runs without an exception. Skipped until the artifacts exist."""
from pathlib import Path

import pandas as pd
import pytest

from churn.config.settings import get_settings
from churn.serving.predictor import PredictionService

ROOT = get_settings().root
APP = ROOT / "app"
REQUIRED = [
    ROOT / "artifacts" / "models" / "final" / "churn_pipeline.joblib",
    ROOT / "artifacts" / "serving" / "background.csv",
    ROOT / "artifacts" / "serving" / "portfolio_validation.csv",
]
pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in REQUIRED),
    reason="Serving artifacts missing (run scripts 07 --evaluate-test and 11)",
)
PAGES = ["main.py", *sorted(f"pages/{p.name}" for p in (APP / "pages").glob("*.py"))]


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "artifacts" / "serving" / name, dtype=str, keep_default_na=False)


def test_service_scores_the_sample_files():
    service = PredictionService.from_artifacts()
    clean = service.predict_batch(_read("sample_batch.csv"))
    assert (clean["status"] == "scored").all()
    broken = service.predict_batch(_read("sample_batch_with_errors.csv"), explain=False)
    assert int((broken["status"] == "rejected").sum()) == 3


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP / page), default_timeout=120)
    app.run()
    assert not app.exception, [e.value for e in app.exception]


def test_prediction_form_submits():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP / "pages" / "2_Churn_Prediction.py"), default_timeout=120)
    app.run()
    submit = [b for b in app.button if b.label == "Predict churn risk"]
    assert submit, "submit button not found"
    submit[0].click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert any(m.label == "Churn probability" for m in app.metric)