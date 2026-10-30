"""Unit tests for the app's Plotly figure builders."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))  # make components importable

from components.charts import (  # noqa: E402
    RAISES,
    LOWERS,
    confusion_figure,
    metric_heatmap,
    reliability_figure,
    roc_pr_figures,
    shap_beeswarm,
    signed_bar,
)


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.3).astype(int)
    p = np.clip(0.4 * y + 0.6 * rng.random(300), 0, 1)
    return y, p


def test_roc_pr_figures_have_one_trace_per_curve(scores):
    y, p = scores
    roc, pr = roc_pr_figures(y, {"A": p, "B": 1 - p})
    assert len(roc.data) == 3          # two curves + the random line
    assert len(pr.data) == 2
    assert "AUC" in roc.data[0].name and "AP" in pr.data[0].name


def test_operating_point_marker(scores):
    y, p = scores
    roc, pr = roc_pr_figures(y, {"A": p}, marker_threshold=0.5)
    assert len(roc.data) == 3          # curve + random line + marker
    assert len(pr.data) == 2           # curve + marker


def test_reliability_figure(scores):
    y, p = scores
    assert len(reliability_figure(y, {"A": p}).data) == 2


def test_confusion_counts():
    fig = confusion_figure([0, 0, 1, 1], [0, 1, 1, 1])
    assert np.asarray(fig.data[0].z).tolist() == [[1, 1], [0, 2]]


def test_metric_heatmap_orders_rows_and_columns():
    summary = pd.DataFrame({"model": ["b", "a", "a", "b"],
                            "strategy": ["none", "none", "smote", "smote"],
                            "roc_auc_mean": [0.80, 0.85, 0.84, 0.79]})
    fig = metric_heatmap(summary, "roc_auc_mean", {"a": "A", "b": "B"},
                         {"none": "No resampling", "smote": "SMOTE"})
    assert list(fig.data[0].y) == ["A", "B"]
    assert list(fig.data[0].x) == ["No resampling", "SMOTE"]


def test_shap_beeswarm_puts_the_largest_feature_first():
    rng = np.random.default_rng(1)
    shap_values = rng.normal(size=(80, 4)) * np.array([0.1, 2.0, 0.5, 0.2])
    fig = shap_beeswarm(shap_values, rng.normal(size=(80, 4)), ["f0", "f1", "f2", "f3"], top_n=3)
    assert list(fig.layout.yaxis.ticktext) == ["f1", "f2", "f3"]


def test_signed_bar_colours():
    fig = signed_bar(["a", "b"], [0.5, -0.2], "x")
    assert list(fig.data[0].marker.color) == [RAISES, LOWERS]