"""Cached loading of the prediction service and saved artifacts.

Everything the app shows comes from files produced by the pipeline scripts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from churn.config.settings import get_settings
from churn.serving.predictor import PredictionService
from churn.utils.logger import get_logger

logger = get_logger("app")


@st.cache_resource(show_spinner="Loading the model...")
def get_service() -> PredictionService | None:
    """One PredictionService shared by every page and every rerun."""
    try:
        return PredictionService.from_artifacts()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return None


def artifact_path(relative: str) -> Path:
    return get_settings().root / relative


@st.cache_data(show_spinner=False)
def load_csv(relative: str) -> pd.DataFrame | None:
    path = artifact_path(relative)
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_json(relative: str) -> dict | None:
    path = artifact_path(relative)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_npz(relative: str) -> dict | None:
    """All arrays in a .npz file. allow_pickle=False: loading never runs code."""
    path = artifact_path(relative)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@st.cache_data(show_spinner=False)
def artifact_bytes(relative: str) -> bytes | None:
    path = artifact_path(relative)
    return path.read_bytes() if path.exists() else None


def portfolio() -> pd.DataFrame | None:
    """Held-out validation customers scored by the final model."""
    frame = load_csv("artifacts/serving/portfolio_validation.csv")
    if frame is None:
        return None
    frame = frame.copy()
    for column in ("segment", "top_driver", "action_1", "action_2", "action_3"):
        if column in frame.columns:
            frame[column] = frame[column].fillna("")
    return frame