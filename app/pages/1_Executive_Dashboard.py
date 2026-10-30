"""Executive Dashboard: portfolio-level churn risk at a glance."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app\components importable

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from churn.analytics.risk import assign_risk_band  # noqa: E402
from components.loaders import load_csv, portfolio  # noqa: E402
from components.ui import (  # noqa: E402
    BAND_COLORS,
    BAND_ORDER,
    page_header,
    require_service,
    setup_page,
    show_chart,
    sidebar_model_card,
    threshold_control,
)

setup_page("Executive Dashboard", "📊")
service = require_service()
threshold = threshold_control(service)
sidebar_model_card(service)

pf = portfolio()
if pf is None:
    page_header("Executive Dashboard")
    st.warning("The scored portfolio is missing. Run: python scripts\\11_prepare_serving.py")
    st.stop()
page_header("Executive Dashboard",
            f"{len(pf):,} held-out validation customers scored by the final model. "
            "Move the decision threshold in the sidebar to see its effect.")

prob = pf["churn_probability"].to_numpy(dtype=float)
bands = pd.Series(assign_risk_band(prob, service.bands_for(threshold)), index=pf.index, name="band")
flagged = prob >= threshold
monthly = pd.to_numeric(pf["MonthlyCharges"], errors="coerce").fillna(0.0).to_numpy()
churned = (pf["Churn"] == "Yes").to_numpy()
revenue_at_risk = prob * monthly

row1 = st.columns(4)
row1[0].metric("Customers", f"{len(pf):,}")
row1[1].metric("Observed churn rate", f"{churned.mean():.1%}")
row1[2].metric("Mean predicted churn", f"{prob.mean():.1%}")
row1[3].metric("Flagged for contact", f"{flagged.sum():,}",
               f"{flagged.mean():.1%} of customers", delta_color="off")
row2 = st.columns(4)
row2[0].metric("Expected churners among flagged", f"{prob[flagged].sum():,.0f}")
row2[1].metric("Monthly revenue at risk", f"{revenue_at_risk.sum():,.0f}")
share = revenue_at_risk[flagged].sum() / max(revenue_at_risk.sum(), 1e-9)
row2[2].metric("...of which in flagged customers", f"{revenue_at_risk[flagged].sum():,.0f}",
               f"{share:.0%} of the total", delta_color="off")
row2[3].metric("Critical-risk customers", f"{(bands == 'Critical').sum():,}")
st.caption("Revenue at risk = sum of churn probability × monthly charge, in the dataset's "
           "currency. It is an expected value, not a forecast of actual loss.")

left, right = st.columns(2)
with left:
    st.subheader("Customers by risk band")
    counts = bands.value_counts().reindex(BAND_ORDER, fill_value=0)
    fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                 color_discrete_map=BAND_COLORS, text=counts.values,
                 labels={"x": "Risk band", "y": "Customers"})
    fig.update_layout(showlegend=False, height=360)
    show_chart(fig)
with right:
    st.subheader("Churn by segment: observed vs predicted")
    if pf["segment"].eq("").all():
        st.info("Segments unavailable. Run scripts\\09_run_segmentation.py and 11_prepare_serving.py.")
    else:
        seg = (pf.assign(Observed=churned * 100.0, Predicted=prob * 100.0)
               .groupby("segment")[["Observed", "Predicted"]].mean().reset_index())
        long = seg.melt(id_vars="segment", var_name="measure", value_name="churn_pct")
        fig = px.bar(long, y="segment", x="churn_pct", color="measure", barmode="group",
                     orientation="h", labels={"churn_pct": "Churn (%)", "segment": ""})
        fig.update_layout(height=360, legend_title_text="")
        show_chart(fig)

left, right = st.columns(2)
with left:
    st.subheader("What drives predicted churn")
    groups = load_csv("artifacts/shap/shap_global_groups.csv")
    if groups is None:
        st.info("Run scripts\\08_explain_model.py to see the drivers.")
    else:
        top = groups.head(10).iloc[::-1]
        fig = px.bar(top, x="mean_abs_shap", y="group", orientation="h",
                     labels={"mean_abs_shap": "Mean |SHAP| (log-odds)", "group": ""})
        fig.update_layout(height=380)
        show_chart(fig)
        st.caption("Shows what the model relies on, not what causes churn.")
with right:
    st.subheader("Risk mix within each segment")
    if not pf["segment"].eq("").all():
        mix = (pd.crosstab(pf["segment"], bands, normalize="index")
               .reindex(columns=BAND_ORDER, fill_value=0) * 100).round(1)
        fig = px.imshow(mix, text_auto=True, aspect="auto", color_continuous_scale="Reds",
                        labels={"color": "% of segment", "x": "Risk band", "y": ""})
        fig.update_layout(height=380)
        show_chart(fig)

with st.expander("Model card"):
    card = service.model_card()
    at = card["at_threshold"]
    st.markdown(
        f"- **Model:** {card['model_label']} ({card['calibration']})\n"
        f"- **Selected by:** {card['selected_by']}: {card['selection_reason']}\n"
        f"- **Default threshold:** {card['threshold']:.2f} ({card['threshold_policy']})\n"
        f"- **Test split at that threshold:** precision {at.get('precision', float('nan')):.3f}, "
        f"recall {at.get('recall', float('nan')):.3f}, "
        f"flagged {at.get('flagged_pct', float('nan')):.1f}% of customers\n"
        f"- **Created:** {card['created_utc']}"
    )