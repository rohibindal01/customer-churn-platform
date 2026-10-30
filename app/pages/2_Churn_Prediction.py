"""Churn Prediction: score one customer, explain the score, suggest actions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app\components importable

import math  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from churn.data import schema  # noqa: E402
from churn.serving.predictor import BatchInputError, consistency_warnings  # noqa: E402
from components.loaders import portfolio  # noqa: E402
from components.ui import (  # noqa: E402
    band_badge,
    disclaimer,
    logit_to_probability,
    page_header,
    require_service,
    setup_page,
    show_chart,
    sidebar_model_card,
    threshold_control,
)

setup_page("Churn Prediction", "🔍")
service = require_service()
threshold = threshold_control(service)
sidebar_model_card(service)
page_header("Churn Prediction",
            "Enter one customer's details. The model estimates the churn probability; "
            "business rules then suggest actions.")

DEFAULT = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 3,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0, "TotalCharges": 255.0,
}
SERVICES = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

options = {"Example: new fiber customer on a monthly contract": DEFAULT}
pf = portfolio()
if pf is not None:
    ranked = pf.sort_values("churn_probability", ascending=False)
    for record in pd.concat([ranked.head(150), ranked.tail(150)]).to_dict(orient="records"):
        label = f"{record['customerID']} (held-out customer, p = {record['churn_probability']:.2f})"
        options[label] = record
choice = st.selectbox("Start from", list(options),
                      help="Pre-fill the form with a held-out customer, then change any field.")
base = options[choice]
key = str(list(options).index(choice))   # new widget keys when the starting point changes


def _number(value, default: float) -> float:
    try:
        number = float(value)
        return default if math.isnan(number) else number
    except (TypeError, ValueError):
        return default


def _select(column: str, label: str | None = None) -> str:
    allowed = list(schema.CATEGORICAL_ALLOWED[column])
    current = str(base.get(column, allowed[0])).strip()
    return st.selectbox(label or column, allowed,
                        index=allowed.index(current) if current in allowed else 0,
                        key=f"{column}_{key}")


with st.form("customer_form"):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Account**")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120,
                                 value=int(_number(base.get("tenure"), 3)), step=1,
                                 key=f"tenure_{key}")
        contract = _select("Contract")
        paperless = _select("PaperlessBilling", "Paperless billing")
        payment = _select("PaymentMethod", "Payment method")
        monthly = st.number_input("Monthly charges", min_value=0.0, max_value=1000.0,
                                  value=_number(base.get("MonthlyCharges"), 70.0), step=0.05,
                                  key=f"monthly_{key}")
        auto_total = st.checkbox("Estimate total charges as tenure × monthly charges",
                                 value=False, key=f"auto_{key}")
        total = st.number_input("Total charges", min_value=0.0, max_value=100000.0,
                                value=_number(base.get("TotalCharges"), 0.0), step=1.0,
                                key=f"total_{key}")
    with col_b:
        st.markdown("**Services**")
        services = {column: _select(column) for column in SERVICES}
    with col_c:
        st.markdown("**Customer**")
        customer = {column: _select(column) for column in ("gender", "Partner", "Dependents")}
        senior = st.selectbox("Senior citizen", ["No", "Yes"],
                              index=1 if _number(base.get("SeniorCitizen"), 0) == 1 else 0,
                              key=f"senior_{key}")
    submitted = st.form_submit_button("Predict churn risk", type="primary")

if not submitted:
    st.info("Fill in the form and press **Predict churn risk**.")
    st.stop()

record = {
    "tenure": tenure, "Contract": contract, "PaperlessBilling": paperless,
    "PaymentMethod": payment, "MonthlyCharges": monthly,
    "TotalCharges": tenure * monthly if auto_total else total,
    **services, **customer, "SeniorCitizen": 1 if senior == "Yes" else 0,
}
for message in consistency_warnings(record):
    st.warning(message)
try:
    with st.spinner("Scoring and explaining..."):
        result = service.predict_one(record, threshold=threshold)
except BatchInputError as exc:
    for message in exc.errors:
        st.error(message)
    st.stop()

cols = st.columns(4)
cols[0].metric("Churn probability", f"{result.probability:.1%}")
cols[1].metric("Decision", "Likely to churn" if result.predicted_churn else "Not flagged",
               f"threshold {result.threshold:.2f}", delta_color="off")
cols[2].markdown(f"**Risk band**<br>{band_badge(result.risk_band)}", unsafe_allow_html=True)
cols[3].markdown(f"**Segment**<br>{result.segment or 'unavailable'}", unsafe_allow_html=True)
st.progress(min(max(result.probability, 0.0), 1.0))
st.caption("The probability is the model's estimate. The decision only says whether it is at or "
           "above the threshold you chose.")

st.subheader("Why this score")
if result.drivers is None:
    st.info("Explanations need artifacts\\serving\\background.csv. "
            "Run scripts\\11_prepare_serving.py.")
else:
    drivers = result.drivers.copy()
    drivers["label"] = [f"{g} = {v}" if v != "" else g
                        for g, v in zip(drivers["group"], drivers["value"])]
    drivers["direction"] = ["raises risk" if s > 0 else "lowers risk" for s in drivers["shap"]]
    fig = px.bar(drivers.iloc[::-1], x="shap", y="label", orientation="h", color="direction",
                 color_discrete_map={"raises risk": "#C0392B", "lowers risk": "#2E86C1"},
                 labels={"shap": f"Contribution ({result.units})", "label": ""})
    fig.update_layout(height=420, legend_title_text="")
    show_chart(fig)
    if result.units == "log-odds":
        st.caption(
            f"An average customer starts at a churn probability of "
            f"{logit_to_probability(result.base_value):.1%}. These contributions move this "
            f"customer to {logit_to_probability(result.model_output):.1%}. They describe what "
            "the model relies on, not what causes churn."
        )

st.subheader("Suggested retention actions")
suggestions = result.recommendations
st.write(suggestions["summary"])
for action in suggestions["actions"]:
    with st.container(border=True):
        st.markdown(f"**{action['title']}** · _{action['basis']}_")
        st.write(action["action"])
        st.caption(f"Why: {action['rationale']}")
        contribution = action["driver_contribution"]
        if contribution is not None:
            verb = "raises" if contribution > 0 else "lowers"
            st.caption(f"{action['driver_family']} {verb} this customer's predicted risk by "
                       f"{abs(contribution):.2f} log-odds.")
disclaimer()