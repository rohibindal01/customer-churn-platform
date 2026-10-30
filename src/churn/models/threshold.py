"""Decision-threshold analysis.

A model outputs a churn probability; a threshold turns it into an action
(contact the customer or not). Thresholds are chosen on validation data,
never on the test split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from churn.config.settings import BusinessSettings

THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
POLICIES = ("default_0.5", "max_f1", "recall_target", "max_net_benefit")


def net_benefit(tp: int, fp: int, business: BusinessSettings) -> float:
    """TP x save_rate x customer_value - (TP + FP) x offer_cost."""
    gained = tp * business.save_rate * business.customer_value
    spent = (tp + fp) * business.offer_cost
    return float(gained - spent)


def breakeven_probability(business: BusinessSettings) -> float:
    """Contact when p x save_rate x customer_value > offer_cost.

    Only meaningful when probabilities are calibrated.
    """
    return business.offer_cost / (business.save_rate * business.customer_value)


def sweep_thresholds(y_true, y_prob, business: BusinessSettings, thresholds=None) -> pd.DataFrame:
    """Confusion counts and metrics at every threshold in the grid."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    grid = THRESHOLD_GRID if thresholds is None else np.asarray(thresholds, dtype=float)
    n = len(y)
    positives = int(y.sum())
    rows = []
    for t in grid:
        flagged = p >= t
        tp = int((flagged & (y == 1)).sum())
        fp = int((flagged & (y == 0)).sum())
        fn = positives - tp
        tn = n - positives - fp
        benefit = net_benefit(tp, fp, business)
        rows.append({
            "threshold": float(t),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
            "recall": tp / positives if positives else float("nan"),
            "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
            "flagged_pct": 100.0 * (tp + fp) / n if n else float("nan"),
            "net_benefit": benefit,
            "net_benefit_per_1000": 1000.0 * benefit / n if n else float("nan"),
        })
    return pd.DataFrame(rows)


def select_thresholds(sweep: pd.DataFrame, recall_target: float) -> pd.DataFrame:
    """One row per policy in POLICIES, taken from the sweep."""
    picks = []

    def take(index, policy: str, note: str = "") -> None:
        row = sweep.loc[index].copy()
        row["policy"] = policy
        row["note"] = note
        picks.append(row)

    take((sweep["threshold"] - 0.5).abs().idxmin(), "default_0.5", "reference only")
    take(sweep["f1"].idxmax(), "max_f1")
    reaching = sweep[sweep["recall"] >= recall_target]
    if reaching.empty:
        take(sweep["recall"].idxmax(), "recall_target",
             f"recall {recall_target:.2f} not reachable; highest recall shown")
    else:
        take(reaching["threshold"].idxmax(), "recall_target",
             f"highest threshold with recall >= {recall_target:.2f}")
    take(sweep["net_benefit"].idxmax(), "max_net_benefit",
         "uses PLACEHOLDER business assumptions")

    out = pd.DataFrame(picks).reset_index(drop=True)
    return out[["policy", *[c for c in out.columns if c != "policy"]]]