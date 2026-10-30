"""SHAP explanations for the final churn model.

The final estimator is clean -> engineer -> preprocess -> model, possibly
inside a calibration wrapper. SHAP explains the inner model on the
preprocessed features, in log-odds units for linear and tree models.
Calibration is monotonic, so it never changes the direction or the order
of the contributions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from churn.features.preprocessing import (
    CATEGORICAL_FEATURES,
    FLAG_FEATURES,
    NUMERIC_FEATURES,
    get_feature_names,
)
from churn.models.final import base_pipeline

LINEAR_MODELS = ("LogisticRegression",)
TREE_MODELS = (
    "GradientBoostingClassifier", "XGBClassifier", "CatBoostClassifier", "LGBMClassifier",
    "RandomForestClassifier", "ExtraTreesClassifier", "DecisionTreeClassifier",
)
_FEATURE_STEPS = ("clean", "engineer")


def feature_groups(feature_names) -> list[str]:
    """Map each preprocessed column to its model feature (one-hot columns -> original feature)."""
    exact = set(NUMERIC_FEATURES) | set(FLAG_FEATURES)
    prefixes = sorted(CATEGORICAL_FEATURES, key=len, reverse=True)  # longest match first
    groups = []
    for name in feature_names:
        if name in exact:
            groups.append(name)
            continue
        match = next((c for c in prefixes if name.startswith(f"{c}_")), None)
        groups.append(match or name)
    return groups


def _positive_class(values) -> np.ndarray:
    """Normalize SHAP output shapes to (rows, features) for the churn class."""
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def _scalar(value) -> float:
    return float(np.ravel(np.asarray(value, dtype=float))[-1])


class ChurnExplainer:
    """SHAP explainer for a fitted churn estimator (pipeline or calibrated pipeline)."""

    def __init__(self, estimator, background: pd.DataFrame, max_background: int = 200,
                 seed: int = 42) -> None:
        import shap  # imported here: it is slow to import and only needed for explanations

        self.pipeline = base_pipeline(estimator)
        self.model = self.pipeline.steps[-1][1]
        self.feature_names = get_feature_names(self.pipeline)
        self.groups = feature_groups(self.feature_names)

        sample = (background.sample(n=max_background, random_state=seed)
                  if len(background) > max_background else background)
        self.background_rows = int(len(sample))
        self.background = self.transform(sample)

        model_type = type(self.model).__name__
        if model_type in LINEAR_MODELS:
            self.kind = "linear"
            self.units = "log-odds"
            self._explainer = shap.LinearExplainer(self.model, self.background)
            self._expected = _scalar(self._explainer.expected_value)
        elif model_type in TREE_MODELS:
            self.kind = "tree"
            self.units = "log-odds"
            self._explainer = shap.TreeExplainer(self.model)
            self._expected = _scalar(self._explainer.expected_value)
        else:
            self.kind = "permutation"
            self.units = "probability"
            self._explainer = shap.Explainer(
                lambda data: self.model.predict_proba(data)[:, 1],
                self.background, algorithm="permutation",
            )
            self._expected = float(self.model.predict_proba(self.background)[:, 1].mean())

    # ---- data views ---------------------------------------------------
    def engineered(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rows after cleaning and feature engineering (readable values)."""
        data = X
        for name, step in self.pipeline.steps:
            if name in _FEATURE_STEPS:
                data = step.transform(data)
        return data

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Rows as the model sees them (after preprocessing; samplers skipped)."""
        data = X
        for name, step in self.pipeline.steps[:-1]:
            if name == "sampler":
                continue
            data = step.transform(data)
        return np.asarray(data, dtype=float)

    # ---- explanations -------------------------------------------------
    @property
    def expected_value(self) -> float:
        """Baseline prediction for the background customers, in self.units."""
        return self._expected

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        data = self.transform(X)
        if self.kind == "permutation":
            return np.asarray(self._explainer(data).values, dtype=float)
        return _positive_class(self._explainer.shap_values(data))

    def model_output(self, X: pd.DataFrame) -> np.ndarray:
        """The inner model's output in the same units as the SHAP values."""
        proba = self.model.predict_proba(self.transform(X))[:, 1]
        if self.units == "probability":
            return proba
        proba = np.clip(proba, 1e-12, 1 - 1e-12)
        return np.log(proba / (1 - proba))


def global_feature_table(shap_values: np.ndarray, X_t: np.ndarray, feature_names,
                         groups) -> pd.DataFrame:
    """Mean |SHAP| and direction of effect for every preprocessed feature."""
    rows = []
    for j, name in enumerate(feature_names):
        col = X_t[:, j]
        values = shap_values[:, j]
        binary = set(np.unique(col)).issubset({0.0, 1.0})
        if binary:
            high, low = col == 1, col == 0
            when = "present"
        else:
            q1, q3 = np.quantile(col, [0.25, 0.75])
            high, low = col >= q3, col <= q1
            when = "high"
        effect_high = float(values[high].mean()) if high.any() else float("nan")
        effect_low = float(values[low].mean()) if low.any() else float("nan")
        if np.isnan(effect_high) or np.isnan(effect_low):
            direction = "not enough variation"
        elif effect_high > effect_low:
            direction = f"raises risk when {when}"
        else:
            direction = f"lowers risk when {when}"
        rows.append({"feature": name, "group": groups[j],
                     "mean_abs_shap": float(np.abs(values).mean()),
                     "mean_shap": float(values.mean()),
                     "effect_high": effect_high, "effect_low": effect_low,
                     "direction": direction})
    table = pd.DataFrame(rows)
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def global_group_table(shap_values: np.ndarray, groups) -> pd.DataFrame:
    """Mean |SHAP| per model feature; one-hot columns are summed BEFORE taking |.|."""
    names = list(dict.fromkeys(groups))
    index = {name: [i for i, g in enumerate(groups) if g == name] for name in names}
    summed = np.column_stack([shap_values[:, index[name]].sum(axis=1) for name in names])
    mean_abs = np.abs(summed).mean(axis=0)
    total = mean_abs.sum()
    table = pd.DataFrame({
        "group": names,
        "mean_abs_shap": mean_abs,
        "share_pct": 100.0 * mean_abs / total if total > 0 else np.zeros(len(names)),
    })
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def _readable(value):
    if isinstance(value, (float, np.floating)):
        return round(float(value), 2)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def explain_customer(explainer: ChurnExplainer, row: pd.DataFrame, top_n: int = 8) -> dict:
    """SHAP contributions for ONE customer, grouped per model feature.

    The contributions (top_n groups plus 'other features') add up exactly to
    model_output - base_value.
    """
    if len(row) != 1:
        raise ValueError("Pass exactly one customer (one row).")
    shap_row = explainer.shap_values(row)[0]
    engineered = explainer.engineered(row).iloc[0]

    grouped = (pd.DataFrame({"group": explainer.groups, "shap": shap_row})
               .groupby("group", sort=False)["shap"].sum().reset_index())
    grouped = grouped.reindex(grouped["shap"].abs().sort_values(ascending=False).index)
    top = grouped.head(top_n)
    rest = float(grouped.iloc[top_n:]["shap"].sum())

    records = [{"group": r.group,
                "value": _readable(engineered[r.group]) if r.group in engineered.index else "",
                "shap": float(r.shap)}
               for r in top.itertuples(index=False)]
    if len(grouped) > top_n:
        records.append({"group": "other features", "value": "", "shap": rest})
    contributions = pd.DataFrame(records)
    if explainer.units == "log-odds":
        contributions["odds_multiplier"] = np.exp(contributions["shap"])

    return {"base_value": explainer.expected_value,
            "model_output": float(explainer.model_output(row)[0]),
            "units": explainer.units,
            "contributions": contributions}


def model_specific_importance(explainer: ChurnExplainer) -> pd.DataFrame:
    """Coefficients (linear models) or built-in importances (tree models)."""
    model, names, groups = explainer.model, explainer.feature_names, explainer.groups
    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        table = pd.DataFrame({"feature": names, "group": groups, "coefficient": coef,
                              "odds_ratio": np.exp(coef), "abs_coefficient": np.abs(coef),
                              "is_zero": coef == 0})
        return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    if hasattr(model, "feature_importances_"):
        table = pd.DataFrame({"feature": names, "group": groups,
                              "importance": np.asarray(model.feature_importances_, dtype=float)})
        return table.sort_values("importance", ascending=False).reset_index(drop=True)
    return pd.DataFrame()


def pick_example_customers(probabilities, threshold: float) -> dict[str, int]:
    """Row positions of illustrative customers: highest, typical flagged, borderline, lowest."""
    p = np.asarray(probabilities, dtype=float)
    order = np.argsort(p)
    examples = {"highest_risk": int(order[-1])}
    flagged = np.where(p >= threshold)[0]
    if flagged.size:
        by_prob = flagged[np.argsort(p[flagged])]
        examples["typical_flagged"] = int(by_prob[len(by_prob) // 2])
        examples["borderline_flagged"] = int(by_prob[0])
    examples["lowest_risk"] = int(order[0])
    unique: dict[str, int] = {}
    for label, idx in examples.items():
        if idx not in unique.values():
            unique[label] = idx
    return unique