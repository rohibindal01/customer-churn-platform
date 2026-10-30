"""Milestone 6B: pre-registered final model selection on VALIDATION data.

The rule is fixed before the test split is opened (see SELECTION_RULE).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

CALIBRATION_TOLERANCE = 0.001
BRIER_TOLERANCE = 0.005
CONFIDENCE = 0.95
SELECTABLE_CALIBRATIONS = ("uncalibrated", "sigmoid")

# Simplest first: linear model, scikit-learn trees, then external boosting libraries.
COMPLEXITY_ORDER = ("logistic_regression", "gradient_boosting", "adaboost", "xgboost", "catboost")

SELECTION_RULE = (
    f"1. Per model, use the uncalibrated pipeline unless sigmoid calibration lowers the "
    f"validation Brier score by more than {CALIBRATION_TOLERANCE}. Isotonic is excluded: in "
    "Milestone 6A it lowered PR-AUC and raised log loss for every model.",
    "2. The reference is the variant with the highest validation PR-AUC.",
    f"3. A variant is eligible when the paired-bootstrap {int(CONFIDENCE * 100)}% interval of "
    "(reference PR-AUC - variant PR-AUC) contains 0 and its Brier score is within "
    f"{BRIER_TOLERANCE} of the best Brier score.",
    "4. Among eligible variants, choose the simplest (complexity order: "
    + " < ".join(COMPLEXITY_ORDER) + "), preferring no calibration wrapper.",
)

_METRICS = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score, "brier": brier_score_loss}


def _resampled_indices(y: np.ndarray, n_boot: int, seed: int):
    """Bootstrap row indices; resamples containing only one class are skipped."""
    rng = np.random.default_rng(seed)
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() != y[idx].max():
            yield idx


def bootstrap_ci(y_true, y_prob, metric: str = "pr_auc", n_boot: int = 2000,
                 seed: int = 42, confidence: float = CONFIDENCE) -> dict:
    """Point estimate and bootstrap confidence interval for one metric."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    fn = _METRICS[metric]
    values = np.array([fn(y[i], p[i]) for i in _resampled_indices(y, n_boot, seed)])
    alpha = (1 - confidence) / 2
    return {"value": float(fn(y, p)),
            "ci_low": float(np.quantile(values, alpha)),
            "ci_high": float(np.quantile(values, 1 - alpha)),
            "n_boot": int(len(values))}


def paired_bootstrap_diff(y_true, prob_a, prob_b, metric: str = "pr_auc", n_boot: int = 2000,
                          seed: int = 42, confidence: float = CONFIDENCE) -> dict:
    """metric(a) - metric(b), computed on the SAME resampled rows each time."""
    y = np.asarray(y_true).astype(int)
    a = np.asarray(prob_a, dtype=float)
    b = np.asarray(prob_b, dtype=float)
    fn = _METRICS[metric]
    diffs = np.array([fn(y[i], a[i]) - fn(y[i], b[i]) for i in _resampled_indices(y, n_boot, seed)])
    alpha = (1 - confidence) / 2
    return {"diff": float(fn(y, a) - fn(y, b)),
            "ci_low": float(np.quantile(diffs, alpha)),
            "ci_high": float(np.quantile(diffs, 1 - alpha)),
            "n_boot": int(len(diffs))}


def candidate_variants(comparison: pd.DataFrame,
                       tolerance: float = CALIBRATION_TOLERANCE) -> pd.DataFrame:
    """Rule step 1: one variant (uncalibrated or sigmoid) per model."""
    usable = comparison[comparison["calibration"].isin(SELECTABLE_CALIBRATIONS)]
    rows = []
    for model, group in usable.groupby("model", sort=False):
        by = group.set_index("calibration")
        has_uncal = "uncalibrated" in by.index
        has_sig = "sigmoid" in by.index
        if has_uncal and (not has_sig
                          or by.loc["sigmoid", "brier"] > by.loc["uncalibrated", "brier"] - tolerance):
            row = by.loc["uncalibrated"].copy()
            row["calibration"] = "uncalibrated"
            row["variant_reason"] = (f"sigmoid did not lower Brier by more than {tolerance}"
                                     if has_sig else "only the uncalibrated variant was available")
        else:
            row = by.loc["sigmoid"].copy()
            row["calibration"] = "sigmoid"
            before = by.loc["uncalibrated", "brier"] if has_uncal else float("nan")
            row["variant_reason"] = f"sigmoid lowered Brier from {before:.4f} to {row['brier']:.4f}"
        row["model"] = model
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def selection_table(variants: pd.DataFrame, val_probs: dict, y_val, n_boot: int = 2000,
                    seed: int = 42, brier_tolerance: float = BRIER_TOLERANCE) -> pd.DataFrame:
    """Rule steps 2 and 3: bootstrap comparison against the reference variant."""
    y = np.asarray(y_val).astype(int)
    variants = variants.reset_index(drop=True)
    keys = [f"{m}__{c}" for m, c in zip(variants["model"], variants["calibration"])]
    missing = [k for k in keys if k not in val_probs]
    if missing:
        raise KeyError(f"No validation probabilities for {missing}")

    reference = keys[int(variants["pr_auc"].to_numpy().argmax())]
    best_brier = float(variants["brier"].min())
    rows = []
    for i, row in variants.iterrows():
        key = keys[i]
        ci = bootstrap_ci(y, val_probs[key], "pr_auc", n_boot, seed)
        if key == reference:
            gap = {"diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        else:
            gap = paired_bootstrap_diff(y, val_probs[reference], val_probs[key], "pr_auc", n_boot, seed)
        tied = gap["ci_low"] <= 0.0 <= gap["ci_high"]
        brier_ok = float(row["brier"]) <= best_brier + brier_tolerance
        model = row["model"]
        rows.append({
            "model": model,
            "calibration": row["calibration"],
            "key": key,
            "pr_auc": ci["value"],
            "pr_auc_ci_low": ci["ci_low"],
            "pr_auc_ci_high": ci["ci_high"],
            "gap_to_reference": gap["diff"],
            "gap_ci_low": gap["ci_low"],
            "gap_ci_high": gap["ci_high"],
            "roc_auc": float(row["roc_auc"]),
            "brier": float(row["brier"]),
            "ece": float(row["ece"]),
            "tied_with_reference": tied,
            "brier_ok": brier_ok,
            "eligible": tied and brier_ok,
            "is_reference": key == reference,
            "complexity_rank": (COMPLEXITY_ORDER.index(model)
                                if model in COMPLEXITY_ORDER else len(COMPLEXITY_ORDER)),
            "wrapper": row["calibration"] != "uncalibrated",
            "variant_reason": row["variant_reason"],
        })
    table = pd.DataFrame(rows)
    return table.sort_values(["complexity_rank", "wrapper"]).reset_index(drop=True)


def select_final(table: pd.DataFrame, model: str | None = None,
                 calibration: str | None = None) -> dict:
    """Rule step 4, or a manual override when model and calibration are given."""
    if model is not None:
        return {"model": model, "calibration": calibration, "selected_by": "manual override",
                "reason": "chosen explicitly with --model and --calibration"}

    reference = table[table["is_reference"]].iloc[0]
    eligible = table[table["eligible"]]
    if eligible.empty:
        return {"model": reference["model"], "calibration": reference["calibration"],
                "selected_by": "rule (fallback)",
                "reason": "no variant met both conditions; using the highest-PR-AUC variant"}

    pick = eligible.sort_values(["complexity_rank", "wrapper"]).iloc[0]
    if pick["key"] == reference["key"]:
        reason = "highest validation PR-AUC and the simplest eligible variant"
    else:
        reason = (f"statistically tied with {reference['key']} (PR-AUC gap "
                  f"{pick['gap_to_reference']:.4f}, 95% CI {pick['gap_ci_low']:.4f} to "
                  f"{pick['gap_ci_high']:.4f}), Brier within tolerance, and simpler")
    return {"model": pick["model"], "calibration": pick["calibration"],
            "selected_by": "rule", "reason": reason}


def format_selection_report(table: pd.DataFrame, decision: dict, policy_rows: pd.DataFrame,
                            chosen_policy: str, business) -> str:
    """Plain-text selection report (validation data only)."""
    view = table[["model", "calibration", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high",
                  "gap_to_reference", "gap_ci_low", "gap_ci_high", "brier",
                  "tied_with_reference", "brier_ok", "eligible"]].round(4)
    pview = policy_rows[["policy", "threshold", "precision", "recall", "f1",
                         "flagged_pct", "net_benefit_per_1000"]].round(3)
    reference = table[table["is_reference"]].iloc[0]["key"]
    lines = [
        "SELECTION RULE (fixed before the test split is opened)",
        *[f"  {line}" for line in SELECTION_RULE],
        "",
        f"VALIDATION EVIDENCE (reference = {reference}; gap = reference PR-AUC - variant PR-AUC)",
        view.to_string(index=False),
        "",
        "VARIANT CHOICE PER MODEL",
        *[f"  {r.model}: {r.calibration} ({r.variant_reason})" for r in table.itertuples(index=False)],
        "",
        f"DECISION: {decision['model']} + {decision['calibration']}  [{decision['selected_by']}]",
        f"  Reason: {decision['reason']}",
        "",
        f"THRESHOLD POLICIES FOR THE CHOSEN MODEL (validation; selected policy = {chosen_policy})",
        pview.to_string(index=False),
        "",
        f"Business costs are PLACEHOLDERS: customer_value={business.customer_value:g}, "
        f"offer_cost={business.offer_cost:g}, save_rate={business.save_rate:g} ({business.currency}).",
    ]
    return "\n".join(lines)