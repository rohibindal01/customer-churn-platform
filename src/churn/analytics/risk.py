"""Risk bands on the calibrated churn probability.

Bands are tied to the thresholds chosen on validation data in Milestone 6:
  Low       p < watch threshold (the recall-target threshold)
  Medium    watch <= p < decision threshold        (watch list)
  High      decision <= p < CRITICAL_PROBABILITY   (contact)
  Critical  p >= CRITICAL_PROBABILITY              (contact first)
The probabilities are calibrated, so 0.70 means roughly 7 in 10 customers
like this one churn.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RISK_BANDS = ("Low", "Medium", "High", "Critical")
CRITICAL_PROBABILITY = 0.70
BAND_ACTIONS = {
    "Low": "No proactive action; standard service.",
    "Medium": "Watch list: low-cost touches only.",
    "High": "Contact with a retention offer.",
    "Critical": "Contact first, with a senior retention agent.",
}


@dataclass(frozen=True)
class BandThresholds:
    watch: float
    decision: float
    critical: float = CRITICAL_PROBABILITY

    def __post_init__(self) -> None:
        if not 0.0 < self.watch <= self.decision < self.critical < 1.0:
            raise ValueError(
                "Need 0 < watch <= decision < critical < 1, got "
                f"watch={self.watch}, decision={self.decision}, critical={self.critical}"
            )


def adjust_decision(thresholds: BandThresholds, decision: float) -> BandThresholds:
    """New decision threshold (for example from the app's slider), keeping bands valid."""
    critical = max(thresholds.critical, min(decision + 0.01, 0.99))
    return BandThresholds(watch=min(thresholds.watch, decision), decision=decision,
                          critical=critical)


def bands_from_metadata(metadata: dict, critical: float = CRITICAL_PROBABILITY) -> BandThresholds:
    """Band thresholds from the final model's metadata.json."""
    decision = float(metadata["threshold"]["value"])
    policies = metadata["threshold"].get("all_policies", {})
    watch = min(float(policies.get("recall_target", decision)), decision)
    return BandThresholds(watch=watch, decision=decision,
                          critical=max(critical, min(decision + 0.01, 0.99)))


def assign_risk_band(probabilities, thresholds: BandThresholds) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    return np.select(
        [p >= thresholds.critical, p >= thresholds.decision, p >= thresholds.watch],
        ["Critical", "High", "Medium"], default="Low",
    )


def band_summary(probabilities, bands, y_true=None) -> pd.DataFrame:
    """Customers, mean predicted and (if labels are given) observed churn per band."""
    p = np.asarray(probabilities, dtype=float)
    b = np.asarray(bands)
    y = None if y_true is None else np.asarray(y_true).astype(int)
    rows = []
    for band in RISK_BANDS:
        mask = b == band
        n = int(mask.sum())
        row = {"band": band, "customers": n, "share_pct": 100.0 * n / len(p) if len(p) else 0.0,
               "mean_predicted_pct": 100.0 * float(p[mask].mean()) if n else float("nan")}
        if y is not None:
            row["observed_churn_pct"] = 100.0 * float(y[mask].mean()) if n else float("nan")
        row["suggested_handling"] = BAND_ACTIONS[band]
        rows.append(row)
    return pd.DataFrame(rows)