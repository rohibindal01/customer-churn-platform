"""Retention Recommendations: a prioritized contact list, the rules and their statistics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app\components importable

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from churn.analytics.recommendations import DISCLAIMER, RULES  # noqa: E402
from churn.analytics.risk import BAND_ACTIONS, RISK_BANDS  # noqa: E402
from components.loaders import get_service, load_csv, portfolio  # noqa: E402
from components.ui import (  # noqa: E402
    page_header,
    require_service,
    setup_page,
    show_chart,
    show_df,
    sidebar_model_card,
    threshold_control,
)

setup_page("Retention Recommendations", "🎯")
service = require_service()
threshold = threshold_control(service)
sidebar_model_card(service)
page_header("Retention Recommendations",
            "Who to contact first, and what to offer. The model provides the probability; "
            "transparent business rules provide the suggestions.")
st.warning(DISCLAIMER)

if portfolio() is None:
    st.warning("The scored portfolio is missing. Run: python scripts\\11_prepare_serving.py")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=10)
def scored_portfolio(threshold: float) -> pd.DataFrame:
    """Re-apply bands and rules to the held-out portfolio at a given threshold."""
    frame = portfolio()
    svc = get_service()
    out = svc.predict_batch(frame[["customerID", *svc.input_columns]], threshold=threshold,
                            explain=True)
    out = out.drop(columns=["row", "status", "problem"]).reset_index(drop=True)
    out["MonthlyCharges"] = pd.to_numeric(frame["MonthlyCharges"], errors="coerce").to_numpy()
    out["tenure"] = frame["tenure"].to_numpy()
    out["Contract"] = frame["Contract"].to_numpy()
    out["actual_outcome"] = np.where(frame["Churn"].to_numpy() == "Yes", "Churned", "Stayed")
    out["expected_revenue_at_risk"] = out["churn_probability"] * out["MonthlyCharges"]
    return out


with st.spinner("Applying the rules to the portfolio..."):
    scored = scored_portfolio(threshold)
flagged = scored[scored["predicted_churn"] == "Yes"]
action_columns = [c for c in scored.columns if c.startswith("action_")]

tab_list, tab_rules, tab_stats = st.tabs(["Contact list", "Rule catalogue", "How often rules fire"])

with tab_list:
    st.caption(f"{len(flagged):,} of {len(scored):,} held-out validation customers are flagged at "
               f"threshold {threshold:.2f}. Choose how many the team can contact and how to rank them.")
    left, right = st.columns([1, 2])
    capacity = left.number_input("Customers the team can contact", min_value=1,
                                 max_value=max(len(flagged), 1),
                                 value=min(100, max(len(flagged), 1)), step=10, key="capacity")
    priority = right.radio("Prioritize by",
                           ["Churn probability", "Expected monthly revenue at risk"],
                           horizontal=True, key="priority")
    sort_column = ("churn_probability" if priority == "Churn probability"
                   else "expected_revenue_at_risk")
    contact = flagged.sort_values(sort_column, ascending=False).head(int(capacity))

    total_risk = max(float(flagged["expected_revenue_at_risk"].sum()), 1e-9)
    covered = float(contact["expected_revenue_at_risk"].sum())
    cols = st.columns(4)
    cols[0].metric("Customers in the contact list", f"{len(contact):,}",
                   f"of {len(flagged):,} flagged", delta_color="off")
    cols[1].metric("Expected churners in the list", f"{contact['churn_probability'].sum():,.0f}")
    cols[2].metric("Monthly revenue at risk covered", f"{covered:,.0f}",
                   f"{covered / total_risk:.0%} of all flagged", delta_color="off")
    cols[3].metric("Actually churned (validation outcome)",
                   f"{(contact['actual_outcome'] == 'Churned').sum():,}",
                   help="Known only because these are held-out validation customers.")

    columns = [c for c in ("customerID", "churn_probability", "risk_band", "segment", "top_driver",
                           *action_columns, "MonthlyCharges", "tenure", "Contract",
                           "actual_outcome") if c in contact.columns]
    show_df(contact[columns].round({"churn_probability": 3, "MonthlyCharges": 2}),
            hide_index=True)
    st.download_button("Download the contact list (CSV)",
                       contact[columns].to_csv(index=False).encode("utf-8"),
                       file_name="retention_contact_list.csv", mime="text/csv", type="primary")

    suggested = pd.concat([contact[c] for c in action_columns], ignore_index=True)
    suggested = suggested[suggested != ""].value_counts()
    if not suggested.empty:
        fig = px.bar(x=suggested.values, y=suggested.index, orientation="h",
                     labels={"x": "Customers", "y": ""},
                     title="Suggestions in this list (top three per customer)")
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=360)
        show_chart(fig)
    st.caption("Each customer's suggestions are ordered by their own risk drivers (SHAP), so the "
               "first suggestion targets what raises that customer's risk the most.")

with tab_rules:
    catalogue = pd.DataFrame([{
        "Rule": r.title, "Driver family": r.family, "Basis": r.basis,
        "Applies to bands": ", ".join(r.bands), "Suggested action": r.action, "Why": r.rationale,
    } for r in RULES])
    show_df(catalogue, hide_index=True)
    config = service.rule_config
    st.caption(f"Settings: new customer = tenure ≤ {config.new_customer_months:g} months; high spend "
               f"= monthly charge ≥ {config.high_spend_monthly:.2f} (training 75th percentile); "
               f"bill increase = current bill above the customer's average by more than "
               f"{config.charge_increase:g}. 'model driver' rules are backed by the explainability "
               "results; 'business heuristic' rules are sensible practice, not model findings.")
    bands = service.bands_for(threshold)
    ranges = {
        "Low": f"p < {bands.watch:.2f}",
        "Medium": f"{bands.watch:.2f} ≤ p < {bands.decision:.2f}",
        "High": f"{bands.decision:.2f} ≤ p < {bands.critical:.2f}",
        "Critical": f"p ≥ {bands.critical:.2f}",
    }
    st.markdown("**Handling per risk band at the current threshold**")
    show_df(pd.DataFrame([{"Band": b, "Probability range": ranges[b], "Handling": BAND_ACTIONS[b]}
                          for b in RISK_BANDS]), hide_index=True)

with tab_stats:
    elevated = scored[scored["risk_band"] != "Low"]
    if elevated.empty:
        st.info("No customers above the Low band at this threshold.")
    else:
        cross = pd.crosstab(elevated["action_1"].replace("", "(no rule matched)"),
                            elevated["risk_band"])
        cross = cross.reindex(columns=[b for b in RISK_BANDS if b in cross.columns])
        st.markdown("**First suggestion per customer, by risk band (current threshold)**")
        show_df(cross)
    static = load_csv("artifacts/recommendations/rule_summary.csv")
    if static is not None:
        with st.expander("Every rule match at the model's default threshold (Milestone 8)"):
            show_df(static, hide_index=True)
    st.caption("Broad rules such as the contract incentive fire for most high-risk customers, so "
               "on their own they don't prioritize. The contact-list capacity and the ordering by "
               "each customer's drivers do.")