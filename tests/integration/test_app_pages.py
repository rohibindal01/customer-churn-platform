"""Interaction tests for pages 6 to 8 on the real artifacts (skipped until they exist)."""
import pytest

from churn.config.settings import get_settings

ROOT = get_settings().root
PAGES = ROOT / "app" / "pages"
REQUIRED = [
    ROOT / "artifacts" / "models" / "final" / "churn_pipeline.joblib",
    ROOT / "artifacts" / "serving" / "portfolio_validation.csv",
    ROOT / "artifacts" / "shap" / "shap_values_validation.npz",
    ROOT / "artifacts" / "metrics" / "benchmark" / "benchmark_summary.csv",
]
pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in REQUIRED),
    reason="Artifacts missing (run scripts 04 to 11)",
)


def _app(name: str):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PAGES / name), default_timeout=180)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def test_benchmark_metric_can_be_switched():
    app = _app("6_Model_Benchmarking.py")
    app.selectbox(key="bench_metric").select("Recall (threshold 0.5)").run()
    assert not app.exception, [e.value for e in app.exception]


def test_another_customer_can_be_explained():
    app = _app("7_Explainable_AI.py")
    box = app.selectbox(key="xai_customer")
    box.select(box.options[-1]).run()
    assert not app.exception, [e.value for e in app.exception]
    assert any(m.label == "Churn probability" for m in app.metric)


def test_contact_list_capacity_changes():
    app = _app("8_Retention_Recommendations.py")
    app.number_input(key="capacity").set_value(25).run()
    assert not app.exception, [e.value for e in app.exception]
    listed = [m for m in app.metric if m.label == "Customers in the contact list"]
    assert listed and listed[0].value == "25"