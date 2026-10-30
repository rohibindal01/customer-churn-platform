"""Explainable AI: what the model relies on, globally and for each customer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app\components importable

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from components.charts import shap_beeswarm, signed_bar  # noqa: E402
from components.loaders import load_csv, load_json, load_npz, portfolio  # noqa: E402
from components.ui import (  # noqa: E402
    band_badge,
    logit_to_probability,
    page_header,
    require_service,
    setup_page,
    show_chart,
    show_df,
    sidebar_model_card,
    threshold_control,
)

setup_page("Explainable AI", "🧠")
service = require_service()
threshold = threshold_control(service)
sidebar_model_card(service)
page_header("Explainable AI",
            "Which features the model relies on, in which direction, and why an individual "
            "customer gets their score.")
st.info("Explanations show what the MODEL relies on, not what CAUSES churn. Changing a feature "
        "for a customer is not proven to change their behaviour without an intervention test.")

meta = load_json("artifacts/shap/explain_meta.json") or {}
groups = load_csv("artifacts/shap/shap_global_groups.csv")
perm = load_csv("artifacts/shap/permutation_importance.csv")
features = load_csv("artifacts/shap/shap_global_features.csv")
coefficients = load_csv("artifacts/shap/model_importance.csv")
shap_data = load_npz("artifacts/shap/shap_values_validation.npz")
if groups is None or shap_data is None:
    st.warning("Explanation artifacts are missing. Run: python scripts\\08_explain_model.py")
    st.stop()
if meta:
    st.caption(f"Explainer: {meta.get('explainer', '?')} ({meta.get('units', '?')}), "
               f"{meta.get('explained_rows', '?')} validation customers, background of "
               f"{meta.get('background_rows', '?')} training customers. Additivity check: "
               f"max error {meta.get('additivity_max_error', float('nan')):.1e}.")

names = [str(n) for n in shap_data["feature_names"]]
values = shap_data["shap_values"]
X_t = shap_data["X_transformed"]

tab_global, tab_summary, tab_coef, tab_customer, tab_dependence = st.tabs(
    ["Global importance", "SHAP summary", "Coefficients", "Explain a customer", "Dependence"])

with tab_global:
    left, right = st.columns(2)
    with left:
        top = groups.head(15)
        fig = px.bar(top, x="mean_abs_shap", y="group", orientation="h",
                     labels={"mean_abs_shap": "Mean |SHAP| (log-odds)", "group": ""},
                     title="SHAP importance per model feature")
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=520)
        show_chart(fig)
        st.caption("One-hot columns are summed per original feature before taking |SHAP|.")
    with right:
        if perm is not None:
            top = perm.head(15)
            fig = px.bar(top, x="importance_mean", y="feature", orientation="h",
                         error_x="importance_std",
                         labels={"importance_mean": "Drop in PR-AUC when shuffled", "feature": ""},
                         title="Permutation importance (raw input columns)")
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=520)
            show_chart(fig)
            st.caption("Shuffling a raw column also scrambles the features engineered from it. "
                       "Correlated inputs (tenure and TotalCharges) share credit.")
    if features is not None:
        st.markdown("**Direction of effect** (mean SHAP when the value is high or present vs "
                    "low or absent)")
        show_df(features[["feature", "mean_abs_shap", "effect_high", "effect_low", "direction"]]
                .head(20).round(4), hide_index=True)
        st.caption("Single features can show surprising signs when correlated features share "
                   "information (for example tenure and tenure_group). Read related features "
                   "together.")

with tab_summary:
    top_n = st.slider("Features shown", 5, 25, 15, key="beeswarm_n")
    show_chart(shap_beeswarm(values, X_t, names, top_n=top_n))
    st.caption("Each dot is one validation customer. Right of zero pushes predicted risk up, "
               "left pushes it down. Red = high value of that feature, blue = low.")

with tab_coef:
    if coefficients is None or coefficients.empty:
        st.info("No model-specific importance available.")
    elif "coefficient" in coefficients.columns:
        nonzero = coefficients[~coefficients["is_zero"]]
        cols = st.columns(3)
        cols[0].metric("Model features", f"{len(coefficients)}")
        cols[1].metric("Non-zero coefficients", f"{len(nonzero)}")
        cols[2].metric("Dropped by the penalty", f"{len(coefficients) - len(nonzero)}")
        top = nonzero.head(20)
        show_chart(signed_bar(top["feature"], top["coefficient"],
                              "Coefficient (change in log-odds; numeric features per 1 SD)"))
        show_df(coefficients[["feature", "coefficient", "odds_ratio", "is_zero"]].round(4),
                hide_index=True)
        st.caption("odds ratio = e^coefficient: 1.5 means 1.5 times the odds of churning. "
                   "Numeric features are standardized. For multi-category features only the "
                   "differences between categories of the same feature are meaningful.")
    else:
        top = coefficients.head(20)
        fig = px.bar(top, x="importance", y="feature", orientation="h",
                     title="Model-specific importance (impurity-based)")
        fig.update_yaxes(autorange="reversed")
        show_chart(fig)

with tab_customer:
    pf = portfolio()
    if pf is None or not service.can_explain:
        st.info("Needs the scored portfolio and the SHAP background "
                "(python scripts\\11_prepare_serving.py).")
    else:
        ranked = pf.sort_values("churn_probability", ascending=False)
        probability_of = dict(zip(ranked["customerID"], ranked["churn_probability"]))
        customer_id = st.selectbox(
            "Validation customer (sorted by churn probability)", ranked["customerID"].tolist(),
            format_func=lambda c: f"{c} (p = {probability_of[c]:.2f})", key="xai_customer")
        row = pf[pf["customerID"] == customer_id].iloc[0]
        result = service.predict_one({c: row[c] for c in service.input_columns},
                                     threshold=threshold, top_n=10)
        cols = st.columns(4)
        cols[0].metric("Churn probability", f"{result.probability:.1%}")
        cols[1].markdown(f"**Risk band**<br>{band_badge(result.risk_band)}",
                         unsafe_allow_html=True)
        cols[2].metric("Actual outcome (validation)",
                       "Churned" if row.get("Churn") == "Yes" else "Stayed")
        cols[3].markdown(f"**Segment**<br>{result.segment or 'unavailable'}",
                         unsafe_allow_html=True)
        drivers = result.drivers.copy()
        drivers["label"] = [f"{g} = {v}" if v != "" else g
                            for g, v in zip(drivers["group"], drivers["value"])]
        show_chart(signed_bar(drivers["label"], drivers["shap"],
                              f"Contribution ({result.units})"))
        if result.units == "log-odds":
            st.caption(f"Average customer: {result.base_value:+.3f} log-odds "
                       f"({logit_to_probability(result.base_value):.1%}). This customer: "
                       f"{result.model_output:+.3f} log-odds "
                       f"({logit_to_probability(result.model_output):.1%}). The contributions "
                       "add up exactly to the difference.")
        columns = [c for c in ("group", "value", "shap", "odds_multiplier") if c in drivers.columns]
        show_df(drivers[columns].round(3), hide_index=True)
        st.caption("To ask 'what if this customer changed contract?', open the customer on the "
                   "Churn Prediction page and edit the form.")

with tab_dependence:
    order = np.argsort(np.abs(values).mean(axis=0))[::-1]
    feature = st.selectbox("Feature", [names[j] for j in order], key="dependence_feature")
    j = names.index(feature)
    points = pd.DataFrame({"feature value": X_t[:, j], "SHAP value": values[:, j],
                           "churn probability": shap_data["probability"]})
    fig = px.scatter(points, x="feature value", y="SHAP value", color="churn probability",
                     color_continuous_scale="Reds", opacity=0.6,
                     title=f"How {feature} moves the prediction")
    fig.add_hline(y=0, line_color="grey")
    fig.update_layout(height=460)
    show_chart(fig)
    st.caption("Numeric features are shown in standardized units (0 = training average). "
               "One-hot features are 0 (absent) or 1 (present). For a linear model the "
               "relationship is a straight line by construction.")