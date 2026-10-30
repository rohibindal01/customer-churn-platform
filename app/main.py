"""Home page of the Customer Churn & Retention Intelligence app.

Run from the project root:
    python -m streamlit run app\\main.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make app\components importable

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components.loaders import artifact_path, get_service  # noqa: E402
from components.ui import page_header, setup_page, show_df, sidebar_model_card  # noqa: E402

setup_page("Home", "📉")
page_header("Customer Churn & Retention Intelligence",
            "Predict which customers are likely to leave, understand why, and decide what to do.")

service = get_service()
if service is None:
    st.error("No trained model was found. Run the pipeline scripts 01 to 11 first "
             "(see the README), then refresh this page.")
    st.stop()
sidebar_model_card(service)
card = service.model_card()


def _metric(ci: dict) -> tuple[str, str]:
    if not ci:
        return "n/a", ""
    return f"{ci['value']:.3f}", f"95% CI {ci['ci_low']:.3f} to {ci['ci_high']:.3f}"


st.subheader("The deployed model")
cols = st.columns(4)
cols[0].metric("Model", card["model_label"])
value, help_text = _metric(card["roc_auc"])
cols[1].metric("Test ROC-AUC", value, help=help_text)
value, help_text = _metric(card["pr_auc"])
cols[2].metric("Test PR-AUC", value, help=help_text)
cols[3].metric("Decision threshold", f"{card['threshold']:.2f}",
               help=f"Policy '{card['threshold_policy']}', chosen on validation data")
st.caption(f"Selected by {card['selected_by']}: {card['selection_reason']}. "
           "Test metrics come from one evaluation on a held-out test split.")

st.subheader("How it works")
st.markdown(
    "1. **Predict:** a logistic regression pipeline estimates each customer's churn probability. "
    "The probabilities are calibrated, so 0.30 means about 3 in 10 such customers leave.\n"
    "2. **Decide:** customers at or above the decision threshold are flagged; risk bands group "
    "them into Low, Medium, High and Critical.\n"
    "3. **Explain:** SHAP shows which features pushed each prediction up or down.\n"
    "4. **Act:** transparent business rules suggest retention actions. They are suggestions, "
    "not model outputs, and their effect has not been measured."
)

st.subheader("Pages")
st.markdown(
    "| Page | Use it to |\n|---|---|\n"
    "| Executive Dashboard | See portfolio risk, revenue at risk and the main drivers |\n"
    "| Churn Prediction | Score one customer and see why |\n"
    "| Batch Prediction | Upload a CSV, validate it and download scores |\n"
    "| Risk Analysis | Explore thresholds, trade-offs and the customers behind them |\n"
    "| Segmentation | Understand customer segments and their churn |\n"
    "| Model Benchmarking | Compare all models, tuning and the final test results |\n"
    "| Explainable AI | Global and individual explanations |\n"
    "| Retention Recommendations | The rule engine and its suggestions |"
)

st.subheader("Artifact status")
checks = {
    "Final model": "artifacts/models/final/churn_pipeline.joblib",
    "Model metadata": "artifacts/models/final/metadata.json",
    "SHAP background": "artifacts/serving/background.csv",
    "Scored portfolio": "artifacts/serving/portfolio_validation.csv",
    "Segment model": "artifacts/segmentation/segment_model.joblib",
    "Recommendation rules": "artifacts/recommendations/recommendation_config.json",
    "Benchmark results": "artifacts/metrics/benchmark/benchmark_summary.csv",
    "SHAP summaries": "artifacts/shap/shap_global_groups.csv",
}
status = pd.DataFrame([{"Artifact": name, "File": rel,
                        "Status": "✅ found" if artifact_path(rel).exists() else "⚠️ missing"}
                       for name, rel in checks.items()])
show_df(status, hide_index=True)