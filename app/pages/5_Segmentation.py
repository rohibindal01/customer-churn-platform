"""Segmentation: who our customers are, and where churn concentrates."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make app\components importable

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from churn.analytics.risk import assign_risk_band  # noqa: E402
from components.loaders import load_csv, load_json, portfolio  # noqa: E402
from components.ui import (  # noqa: E402
    BAND_COLORS,
    BAND_ORDER,
    page_header,
    require_service,
    setup_page,
    show_chart,
    show_df,
    sidebar_model_card,
    threshold_control,
)

setup_page("Segmentation", "🧩")
service = require_service()
threshold = threshold_control(service)
sidebar_model_card(service)
page_header("Customer Segmentation",
            "K-Means segments built from customer attributes only. The churn label was used "
            "afterwards to profile them, never to build them.")

meta = load_json("artifacts/segmentation/segmentation_meta.json")
profile = load_csv("artifacts/segmentation/segment_profile_train.csv")
if meta is None or profile is None:
    st.warning("Segmentation results are missing. Run: python scripts\\09_run_segmentation.py")
    st.stop()
choice = meta["choice"]
k_table = load_csv("artifacts/segmentation/k_selection.csv")
stability = (float(k_table.loc[k_table["k"] == meta["k"], "stability_ari"].iloc[0])
             if k_table is not None and (k_table["k"] == meta["k"]).any() else float("nan"))

cols = st.columns(4)
cols[0].metric("Segments (k)", meta["k"])
cols[1].metric("Silhouette", f"{choice['silhouette']:.3f}",
               help="Above 0.5 strong, 0.25 to 0.5 reasonable, below 0.25 weak structure")
cols[2].metric("Cluster structure", choice["structure"].capitalize())
cols[3].metric("Bootstrap stability (ARI)", f"{stability:.2f}",
               help="1.0 = identical segments every time the data is resampled")
st.caption(f"Why k = {meta['k']}: {choice['reason']}.")
if choice["structure"] == "weak":
    st.warning("Weak structure: treat these segments as a practical partition of a continuum, "
               "not as naturally separate groups.")

tab_profiles, tab_map, tab_choice, tab_explore = st.tabs(
    ["Profiles", "Segment map", "Choosing k", "Explore a segment"])

with tab_profiles:
    overall = float((profile["churn_rate_pct"] * profile["customers"]).sum()
                    / profile["customers"].sum())
    fig = px.bar(profile.sort_values("churn_rate_pct"), x="churn_rate_pct", y="name",
                 orientation="h", text="customers",
                 labels={"churn_rate_pct": "Churn rate (%)", "name": ""},
                 title="Churn rate by segment (training split; bar label = customers)")
    fig.add_vline(x=overall, line_dash="dash", line_color="#C0392B",
                  annotation_text=f"overall {overall:.1f}%")
    fig.update_layout(height=380)
    show_chart(fig)
    columns = [c for c in ("name", "customers", "share_pct", "tenure_mean", "monthly_mean",
                           "addons_mean", "month_to_month_pct", "fiber_pct", "no_internet_pct",
                           "no_security_or_support_pct", "churn_rate_pct", "lift",
                           "mean_predicted_pct") if c in profile.columns]
    show_df(profile[columns].round(2), hide_index=True)
    st.caption("lift = segment churn ÷ overall churn. mean_predicted_pct = the final model's "
               "average churn probability in the segment.")
    stable = load_csv("artifacts/segmentation/segment_stability.csv")
    if stable is not None:
        st.markdown("**Do the profiles hold on unseen customers?** (validation split)")
        show_df(stable.round(2), hide_index=True)
    st.caption("Segments describe customers; they are not causes of churn. A segment's churn "
               "rate mixes the effects of all its traits.")

with tab_map:
    pf = portfolio()
    if pf is None or service.segment_model is None:
        st.info("The map needs the scored portfolio and the segment model "
                "(scripts 09 and 11).")
    else:
        scaled = service.segment_model.transform(pf[service.input_columns])
        pca = PCA(n_components=2, random_state=42).fit(scaled)
        coords = pca.transform(scaled)
        points = pd.DataFrame({
            "PC1": coords[:, 0], "PC2": coords[:, 1], "segment": pf["segment"],
            "customerID": pf["customerID"], "churn probability": pf["churn_probability"].round(3),
            "tenure": pf["tenure"], "MonthlyCharges": pf["MonthlyCharges"],
        })
        ratio = pca.explained_variance_ratio_
        fig = px.scatter(points, x="PC1", y="PC2", color="segment", opacity=0.6,
                         hover_data=["customerID", "churn probability", "tenure",
                                     "MonthlyCharges"],
                         labels={"PC1": f"PC1 ({ratio[0]:.0%} of variance)",
                                 "PC2": f"PC2 ({ratio[1]:.0%} of variance)"})
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(height=560, legend_title_text="")
        show_chart(fig)
        st.caption(f"{len(points):,} validation customers projected from the 8 segmentation "
                   "inputs to 2 dimensions with PCA. Overlap in 2D does not always mean overlap "
                   "in the full 8 dimensions. Hover a dot to see the customer.")

with tab_choice:
    if k_table is not None:
        left, right = st.columns(2)
        with left:
            fig = px.line(k_table, x="k", y="inertia", markers=True,
                          title="Elbow: within-cluster sum of squares")
            fig.add_vline(x=meta["k"], line_dash="dash", line_color="#C0392B")
            show_chart(fig)
        with right:
            long = k_table.melt(id_vars="k", value_vars=["silhouette", "stability_ari"],
                                var_name="measure", value_name="value")
            fig = px.line(long, x="k", y="value", color="measure", markers=True,
                          title="Silhouette and bootstrap stability")
            fig.add_vline(x=meta["k"], line_dash="dash", line_color="#C0392B")
            fig.add_hline(y=0.25, line_dash="dot", line_color="#7F8C8D")
            fig.add_hline(y=0.80, line_dash="dot", line_color="#7F8C8D")
            show_chart(fig)
        show_df(k_table.round(4), hide_index=True)
        st.caption("Rule fixed before looking: k from 3 to 8, stability ≥ 0.80, every segment "
                   "≥ 5% of customers, then the highest silhouette.")
    methods = load_csv("artifacts/segmentation/method_comparison.csv")
    if methods is not None:
        st.markdown("**Does another algorithm find the same segments?**")
        show_df(methods.round(4), hide_index=True)
        st.caption("ARI vs K-Means: 1 = identical grouping, 0 = no agreement beyond chance. "
                   "DBSCAN's silhouette excludes the points it marks as noise.")

with tab_explore:
    pf = portfolio()
    chosen = st.selectbox("Segment", profile["name"].tolist(), key="segment_choice")
    row = profile[profile["name"] == chosen].iloc[0]
    cols = st.columns(4)
    cols[0].metric("Customers (training)", f"{int(row['customers']):,}",
                   f"{row['share_pct']:.1f}% of all", delta_color="off")
    cols[1].metric("Churn rate", f"{row['churn_rate_pct']:.1f}%")
    cols[2].metric("Lift vs overall", f"{row['lift']:.2f}×")
    if "mean_predicted_pct" in profile.columns:
        cols[3].metric("Mean predicted churn", f"{row['mean_predicted_pct']:.1f}%")
    if pf is not None and (pf["segment"] == chosen).any():
        members = pf[pf["segment"] == chosen]
        bands = pd.Series(assign_risk_band(members["churn_probability"].to_numpy(dtype=float),
                                           service.bands_for(threshold)))
        left, right = st.columns(2)
        with left:
            counts = bands.value_counts().reindex(BAND_ORDER, fill_value=0)
            fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                         color_discrete_map=BAND_COLORS, text=counts.values,
                         labels={"x": "Risk band", "y": "Validation customers"},
                         title="Risk bands at the current threshold")
            fig.update_layout(showlegend=False, height=340)
            show_chart(fig)
        with right:
            drivers = members["top_driver"].replace("", "none raising risk").value_counts()
            fig = px.bar(x=drivers.values, y=drivers.index, orientation="h",
                         labels={"x": "Customers", "y": ""},
                         title="Each customer's largest risk driver")
            fig.update_layout(height=340)
            show_chart(fig)
        compare = pf.assign(group=["This segment" if s == chosen else "All other customers"
                                   for s in pf["segment"]])
        left, right = st.columns(2)
        for column, holder in (("tenure", left), ("MonthlyCharges", right)):
            with holder:
                fig = px.histogram(compare, x=column, color="group", barmode="overlay",
                                   histnorm="percent", nbins=30, opacity=0.6)
                fig.update_layout(height=300, legend_title_text="")
                show_chart(fig)