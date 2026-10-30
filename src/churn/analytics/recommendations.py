"""Interpretable retention recommendation engine.

PREDICTIONS come from the model. RECOMMENDATIONS come from the transparent
business rules below. A rule never changes a probability. Rules use the
customer's attributes and risk band; suggestions are ordered by which
driver families raise THIS customer's predicted risk (from SHAP).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from churn.analytics.risk import RISK_BANDS
from churn.features.cleaning import clean_features
from churn.features.engineering import engineer_features

DISCLAIMER = (
    "Recommendations are rule-based suggestions, separate from the model's prediction. "
    "Their effect on churn has not been measured; test them, for example with a controlled "
    "A/B test, before rolling them out."
)
BASES = ("model driver", "business heuristic")

DRIVER_FAMILIES = {
    "Tenure": ("tenure", "tenure_group"),
    "Spend": ("MonthlyCharges", "TotalCharges", "avg_monthly_spend", "charge_gap"),
    "Contract": ("Contract",),
    "Internet service": ("InternetService",),
    "Security and support": ("no_security_or_support", "OnlineSecurity", "TechSupport"),
    "Add-on services": ("addon_count", "OnlineBackup", "DeviceProtection",
                        "StreamingTV", "StreamingMovies"),
    "Billing and payment": ("PaperlessBilling", "PaymentMethod", "is_auto_payment"),
    "Phone service": ("PhoneService", "MultipleLines"),
    "Household": ("gender", "SeniorCitizen", "Partner", "Dependents"),
}
_GROUP_TO_FAMILY = {group: family for family, groups in DRIVER_FAMILIES.items()
                    for group in groups}


def family_of(group: str) -> str:
    return _GROUP_TO_FAMILY.get(group, "Other")


def family_contributions(shap_values, groups) -> pd.DataFrame:
    """SHAP summed per driver family, one row per customer."""
    values = np.asarray(shap_values, dtype=float)
    families = [family_of(g) for g in groups]
    names = [f for f in [*DRIVER_FAMILIES, "Other"] if f in families]
    return pd.DataFrame({name: values[:, [i for i, f in enumerate(families) if f == name]]
                         .sum(axis=1) for name in names})


def customer_records(X_raw: pd.DataFrame) -> list[dict]:
    """Cleaned and engineered attributes per customer, as dictionaries for the rules."""
    return engineer_features(clean_features(X_raw)).to_dict(orient="records")


@dataclass(frozen=True)
class RuleConfig:
    new_customer_months: float = 12.0
    high_spend_monthly: float = 90.0     # the script sets the training 75th percentile
    charge_increase: float = 5.0
    min_band_for_action: str = "Medium"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    title: str
    family: str
    basis: str
    priority: int
    bands: tuple
    condition: Callable[[dict, RuleConfig], bool]
    action: str
    rationale: str


_MEDIUM_UP = ("Medium", "High", "Critical")
_HIGH_UP = ("High", "Critical")

RULES = (
    Rule("contract_incentive", "Contract incentive", "Contract", "model driver", 1, _HIGH_UP,
         lambda c, cfg: c["Contract"] == "Month-to-month",
         "Offer a discounted one- or two-year contract, for example a price lock.",
         "In the final model, month-to-month contracts raise predicted risk and two-year "
         "contracts lower it."),
    Rule("onboarding", "New-customer onboarding", "Tenure", "model driver", 1, _MEDIUM_UP,
         lambda c, cfg: c["tenure"] <= cfg.new_customer_months,
         "Schedule an onboarding check-in: service setup, first-bill walkthrough, satisfaction call.",
         "Short tenure is the final model's largest single risk driver."),
    Rule("security_support_trial", "Security or support trial", "Security and support",
         "model driver", 2, _MEDIUM_UP,
         lambda c, cfg: c["no_security_or_support"] == 1,
         "Offer a free trial of Online Security or Tech Support.",
         "Internet customers with neither security nor support carry higher predicted risk."),
    Rule("fiber_service_review", "Fiber service review", "Internet service", "model driver", 3,
         _HIGH_UP, lambda c, cfg: c["InternetService"] == "Fiber optic",
         "Run a proactive service-quality check and review whether the price fits the plan.",
         "Fiber raises predicted risk. The data cannot say whether price or quality drives "
         "this, so investigate before offering discounts."),
    Rule("high_value_loyalty", "Protect a high-value customer", "Spend", "business heuristic", 2,
         _HIGH_UP, lambda c, cfg: c["MonthlyCharges"] >= cfg.high_spend_monthly,
         "Offer a loyalty benefit or priority service; involve a senior retention agent.",
         "High monthly revenue is at risk, so a successful save is worth more."),
    Rule("bill_increase_review", "Review a recent bill increase", "Spend", "business heuristic",
         3, _MEDIUM_UP, lambda c, cfg: c["charge_gap"] > cfg.charge_increase,
         "Review the plan: the current bill is above the customer's own historical average.",
         "A bill rising above a customer's usual spend can prompt comparison shopping."),
    Rule("autopay_nudge", "Move to automatic payment", "Billing and payment",
         "business heuristic", 5, _MEDIUM_UP,
         lambda c, cfg: c["PaymentMethod"] == "Electronic check",
         "Encourage automatic payment (bank transfer or card), for example with a small credit.",
         "Low-cost friction reduction. Payment method adds little to the final model's "
         "predictions, so this is not a model finding."),
    Rule("senior_assist", "Assisted support", "Household", "business heuristic", 4, _HIGH_UP,
         lambda c, cfg: c["SeniorCitizen"] == 1,
         "Offer an assisted support channel, such as a dedicated phone line or in-home setup.",
         "A service-quality suggestion, not a model finding."),
)


def _sort_key(item: dict):
    c = item["driver_contribution"]
    raises = c is not None and c > 0
    return (0 if raises else 1, -c if raises else 0.0, item["priority"])


def recommend(customer: dict, probability: float, band: str, contributions: dict | None = None,
              config: RuleConfig = RuleConfig(), max_items: int = 3) -> dict:
    """Rule-based retention suggestions for ONE customer."""
    if band not in RISK_BANDS:
        raise ValueError(f"Unknown risk band '{band}'. Choose from {RISK_BANDS}.")
    base = {"probability": float(probability), "band": band, "disclaimer": DISCLAIMER}
    if RISK_BANDS.index(band) < RISK_BANDS.index(config.min_band_for_action):
        return {**base, "actions": [], "matched_rule_ids": [],
                "summary": "Low risk: no proactive retention action suggested; standard service."}

    matched = []
    for rule in RULES:
        if band not in rule.bands:
            continue
        try:
            fired = bool(rule.condition(customer, config))
        except (KeyError, TypeError, ValueError):
            fired = False
        if not fired:
            continue
        contribution = None if contributions is None else contributions.get(rule.family)
        if contribution is not None and np.isnan(float(contribution)):
            contribution = None
        matched.append({
            "rule_id": rule.rule_id, "title": rule.title, "action": rule.action,
            "rationale": rule.rationale, "basis": rule.basis, "driver_family": rule.family,
            "driver_contribution": None if contribution is None else float(contribution),
            "priority": rule.priority,
        })
    matched.sort(key=_sort_key)
    summary = (f"{len(matched)} suggestion(s) for a {band.lower()}-risk customer."
               if matched else "Elevated risk but no rule matched: review this customer manually.")
    return {**base, "actions": matched[:max_items],
            "matched_rule_ids": [m["rule_id"] for m in matched], "summary": summary}