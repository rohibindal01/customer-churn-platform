"""Risk Analysis: thresholds, trade-offs and the customers behind the numbers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app\components importable

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from churn.analytics.risk import assign_risk_band, band_summary  # noqa: E402
from churn.config.settings import BusinessSettings, get_settings  # noqa: E402
from churn.models.final import metrics_at_threshold  # noqa: E402
from churn.models.threshold import breakeven_probability, sweep_thresholds  # noqa: E402
from components.loaders import portfolio  # noqa: E402
from components.ui import (  # noqa: E402
    BAND_ORDER,
    page_header,
    placeholder_note,
    require_service,
    setup_page,
    show_chart,
    show_df,
    sidebar_model_card,
    threshold_control,
)

setup_page("Risk Analysis", "⚖️")
service = require_service()
threshold = threshold_control(service)
defaults = get_settings().business
with st.sidebar.expander("Business assumptions (placeholders)"):
    value = st.number_input("Value of a retained customer", min_value=1.0,
                            value=float(defaults.customer_value), step=50.0)
    offer = st.number_input("Cost of one retention offer", min_value=0.0,
                            value=float(defaults.offer_cost), step=5.0)
    save = st.slider("Save rate (share of contacted churners kept)", 0.01, 1.0,
                     float(defaults.save_rate), 0.01)
business = BusinessSettings(customer_value=value, offer_cost=offer, save_rate=save,
                            currency=defaults.currency)
sidebar_model_card(service)
page_header("Risk Analysis",
            "How the threshold trades caught churners against false alarms, on held-out "
            "validation customers with known outcomes.")

pf = portfolio()
if pf is None:
    st.warning("The scored portfolio is missing. Run: python scripts\\11_prepare_serving.py")
    st.stop()
y = (pf["Churn"] == "Yes").astype(int).to_numpy()
prob = pf["churn_probability"].to_numpy(dtype=float)

m = metrics_at_threshold(y, prob, threshold, business)
cols = st.columns(5)
cols[0].metric("Precision", f"{m['precision']:.1%}", help="Share of flagged customers who churned")
cols[1].metric("Recall", f"{m['recall']:.1%}", help="Share of churners who were flagged")
cols[2].metric("Flagged", f"{m['flagged_pct']:.1f}%")
cols[3].metric("Net benefit per 1,000 customers", f"{m['net_benefit_per_1000']:,.0f}")
cols[4].metric("Break-even probability", f"{breakeven_probability(business):.2f}",
               help="offer_cost / (save_rate × customer_value): contacting pays off above this "
                    "probability, if the probabilities are calibrated")
st.caption(f"At threshold {threshold:.2f}: {m['tp']} churners caught, {m['fp']} false alarms, "
           f"{m['fn']} churners missed, {m['tn']} customers correctly left alone.")
placeholder_note()

sweep = sweep_thresholds(y, prob, business)
left, right = st.columns(2)
with left:
    long = sweep.melt(id_vars="threshold", value_vars=["precision", "recall", "f1"],
                      var_name="metric", value_name="value")
    fig = px.line(long, x="threshold", y="value", color="metric",
                  labels={"threshold": "Threshold", "value": ""},
                  title="Precision, recall and F1")
    fig.add_vline(x=threshold, line_dash="dash", line_color="#C0392B")
    fig.update_layout(height=380, legend_title_text="")
    show_chart(fig)
with right:
    fig = px.line(sweep, x="threshold", y="net_benefit_per_1000",
                  labels={"threshold": "Threshold",
                          "net_benefit_per_1000": f"Net benefit per 1,000 ({business.currency})"},
                  title="Net benefit (placeholder costs)")
    fig.add_vline(x=threshold, line_dash="dash", line_color="#C0392B")
    fig.add_vline(x=breakeven_probability(business), line_dash="dot", line_color="#7F8C8D")
    fig.add_hline(y=0, line_color="#7F8C8D", line_width=1)
    fig.update_layout(height=380)
    show_chart(fig)

st.subheader("Risk bands: predicted vs observed")
bands = assign_risk_band(prob, service.bands_for(threshold))
show_df(band_summary(prob, bands, y).round(1), hide_index=True)
st.caption("If the model is well calibrated, observed churn sits close to the mean predicted value "
           "in every band.")

st.subheader("Customer explorer")
view = pf.assign(band=bands)
f1, f2, f3 = st.columns(3)
chosen_bands = f1.multiselect("Risk bands", BAND_ORDER, default=["High", "Critical"])
segments = sorted(s for s in view["segment"].unique() if s)
chosen_segments = f2.multiselect("Segments", segments, default=segments)
min_prob = f3.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.01)
mask = view["band"].isin(chosen_bands) & (view["churn_probability"] >= min_prob)
if segments:
    mask &= view["segment"].isin(chosen_segments)
columns = [c for c in ("customerID", "churn_probability", "band", "segment", "top_driver",
                       "action_1", "tenure", "Contract", "MonthlyCharges", "Churn")
           if c in view.columns]
table = view.loc[mask, columns].sort_values("churn_probability", ascending=False)
st.caption(f"{len(table):,} customers match the filters.")
show_df(table, hide_index=True)
st.download_button("Download this list (CSV)", table.to_csv(index=False).encode("utf-8"),
                   file_name="risk_list.csv", mime="text/csv")
st.caption("These are the validation customers used to choose the thresholds, so the figures are "
           "slightly optimistic. The one-time test results are on the Model Benchmarking page.")