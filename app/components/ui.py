"""Shared page setup, sidebar controls and small display helpers."""
from __future__ import annotations

import math

import streamlit as st

from churn.analytics.recommendations import DISCLAIMER
from components.loaders import get_service

APP_NAME = "Churn Intelligence"
BAND_ORDER = ["Low", "Medium", "High", "Critical"]
BAND_COLORS = {"Low": "#4C9A2A", "Medium": "#D4A017", "High": "#E07A1F", "Critical": "#C0392B"}

_CSS = """
<style>
.block-container {padding-top: 1.8rem;}
.band {display:inline-block; padding:0.15rem 0.7rem; border-radius:0.9rem;
       color:white; font-weight:600; font-size:0.9rem;}
</style>
"""


def setup_page(title: str, icon: str = "📉") -> None:
    st.set_page_config(page_title=f"{title} | {APP_NAME}", page_icon=icon, layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def require_service():
    service = get_service()
    if service is None:
        st.error("No trained model was found in artifacts\\models\\final. Run the pipeline "
                 "scripts 01 to 11 first; the README lists them in order.")
        st.stop()
    return service


def threshold_control(service) -> float:
    """Sidebar threshold slider shared by all pages through session_state."""
    default = float(service.threshold)
    if "threshold_value" not in st.session_state:
        st.session_state["threshold_value"] = default

    def _sync() -> None:
        st.session_state["threshold_value"] = float(st.session_state["_threshold_slider"])

    st.sidebar.slider(
        "Decision threshold", min_value=0.05, max_value=0.95,
        value=float(st.session_state["threshold_value"]), step=0.01,
        key="_threshold_slider", on_change=_sync,
        help="Customers with a churn probability at or above this value are flagged for contact.",
    )
    info = service.metadata.get("threshold", {})
    presets = ", ".join(f"{name} {value:.2f}" for name, value in info.get("all_policies", {}).items()
                        if name != "default_0.5")
    st.sidebar.caption(f"Model default {default:.2f} ({info.get('policy', '')}). "
                       f"Validation presets: {presets}.")
    if st.sidebar.button("Reset to model default"):
        st.session_state["threshold_value"] = default
        st.session_state.pop("_threshold_slider", None)
        st.rerun()
    return float(st.session_state["threshold_value"])


def sidebar_model_card(service) -> None:
    card = service.model_card()
    roc, pr = card["roc_auc"].get("value"), card["pr_auc"].get("value")
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**Model:** {card['model_label']} ({card['calibration']})")
        if roc is not None and pr is not None:
            st.caption(f"One-time test evaluation: ROC-AUC {roc:.3f}, PR-AUC {pr:.3f}")


def band_badge(band: str) -> str:
    color = BAND_COLORS.get(band, "#7F8C8D")
    return f'<span class="band" style="background:{color}">{band}</span>'


def show_df(frame, **kwargs) -> None:
    """st.dataframe across Streamlit versions (width='stretch' replaced use_container_width)."""
    try:
        st.dataframe(frame, width="stretch", **kwargs)
    except Exception:  # noqa: BLE001 - older Streamlit versions
        st.dataframe(frame, use_container_width=True, **kwargs)


def show_chart(figure) -> None:
    try:
        st.plotly_chart(figure, width="stretch")
    except Exception:  # noqa: BLE001 - older Streamlit versions
        st.plotly_chart(figure, use_container_width=True)


def disclaimer() -> None:
    st.caption(f"⚠️ {DISCLAIMER}")


def placeholder_note() -> None:
    st.caption("Net-benefit figures use PLACEHOLDER business costs. They illustrate the method, "
               "not real revenue. Change them in the sidebar or in config.yaml.")


def logit_to_probability(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))