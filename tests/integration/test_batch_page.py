"""The Batch Prediction page end to end, driven through its built-in sample sources.

AppTest cannot simulate a browser file upload, but samples feed the same bytes into
the same code path, so the whole results view runs here.
"""
import pytest

from churn.config.settings import get_settings

ROOT = get_settings().root
PAGE = ROOT / "app" / "pages" / "3_Batch_Prediction.py"
REQUIRED = [
    ROOT / "artifacts" / "models" / "final" / "churn_pipeline.joblib",
    ROOT / "artifacts" / "serving" / "sample_batch.csv",
    ROOT / "artifacts" / "serving" / "sample_batch_with_errors.csv",
]
pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in REQUIRED),
    reason="Serving artifacts missing (run scripts 07 --evaluate-test and 11)",
)
VALID = "Sample: 25 valid customers"
INVALID = "Sample: 5 rows, 3 deliberately invalid"


def _page():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PAGE), default_timeout=180)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def _metric(app, label: str) -> str:
    return next(m.value for m in app.metric if m.label == label)


def _choose(app, source: str):
    app.radio(key="batch_source").set_value(source).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def test_upload_is_the_default_and_waits_for_a_file():
    app = _page()
    assert app.radio(key="batch_source").value == "Upload a CSV file"
    assert any("Upload a file to start" in info.value for info in app.info)
    assert not app.metric


def test_valid_sample_is_fully_scored():
    app = _choose(_page(), VALID)
    assert (_metric(app, "Rows"), _metric(app, "Scored"), _metric(app, "Rejected")) == \
        ("25", "25", "0")
    assert not app.warning
    assert len(app.dataframe) >= 1


def test_invalid_rows_are_rejected_with_reasons():
    app = _choose(_page(), INVALID)
    assert (_metric(app, "Rows"), _metric(app, "Scored"), _metric(app, "Rejected")) == \
        ("5", "2", "3")
    assert any("could not be scored" in warning.value for warning in app.warning)
    assert len(app.dataframe) >= 2       # rejected-rows table and results table


def test_scoring_without_explanations():
    app = _choose(_page(), VALID)
    app.checkbox(key="batch_explain").uncheck().run()
    assert not app.exception, [e.value for e in app.exception]
    assert _metric(app, "Scored") == "25"