"""PredictionService: the single entry point for the app (and any future API).

It loads saved artifacts once and turns raw customer rows into:
probability -> decision at a threshold -> risk band -> segment ->
SHAP drivers -> rule-based suggestions. It never trains anything.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from churn.analytics.recommendations import (
    RuleConfig,
    customer_records,
    family_contributions,
    recommend,
)
from churn.analytics.risk import (
    BandThresholds,
    adjust_decision,
    assign_risk_band,
    bands_from_metadata,
)
from churn.config.settings import Settings, get_settings
from churn.data import schema
from churn.models.final import FINAL_MODEL_FILE, METADATA_FILE
from churn.utils.logger import get_logger

logger = get_logger(__name__)

MAX_BATCH_ROWS = 100_000
BACKGROUND_FILE = "background.csv"
ADDON_COLUMNS = ("OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies")


class BatchInputError(ValueError):
    """Input that cannot be scored at all; carries readable messages."""

    def __init__(self, errors) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


# ---------------------------------------------------------------------
# Input checks and normalisation
# ---------------------------------------------------------------------
def _text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _blank(series: pd.Series) -> pd.Series:
    text = _text(series)
    return (text.isna() | (text == "").fillna(False)).astype(bool)


def _number(series: pd.Series) -> pd.Series:
    text = _text(series)
    text = text.mask((text == "").fillna(False))
    values = pd.to_numeric(text, errors="coerce")
    return pd.Series(values.to_numpy(dtype="float64", na_value=np.nan), index=series.index)


def row_problems(df: pd.DataFrame) -> pd.Series:
    """One message per row: '' means the row can be scored.

    Required columns must already exist (see PredictionService.check_file).
    A blank TotalCharges is allowed: the pipeline handles new customers.
    """
    messages: list[list[str]] = [[] for _ in range(len(df))]

    def add(mask, message: str) -> None:
        for i in np.flatnonzero(np.asarray(mask, dtype=bool)):
            messages[i].append(message)

    for column, allowed in schema.CATEGORICAL_ALLOWED.items():
        blank = _blank(df[column])
        known = _text(df[column]).isin(list(allowed)).fillna(False).astype(bool)
        add(blank, f"{column} is missing")
        add(~blank & ~known, f"{column} has an unknown value")

    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        blank = _blank(df[column])
        number = _number(df[column])
        if column != "TotalCharges":
            add(blank, f"{column} is missing")
        add(~blank & number.isna(), f"{column} is not a number")
        add((number < 0).fillna(False), f"{column} is negative")

    add(~_number(df["SeniorCitizen"]).isin([0.0, 1.0]), "SeniorCitizen must be 0 or 1")
    return pd.Series(["; ".join(m) for m in messages], index=df.index, dtype=object)


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    """The 19 model inputs with whitespace stripped and numbers parsed."""
    out = df[list(schema.RAW_FEATURE_COLUMNS)].copy()
    for column in schema.CATEGORICAL_COLUMNS:
        out[column] = _text(out[column]).astype(object)
    for column in ("tenure", "MonthlyCharges", "SeniorCitizen"):
        out[column] = _number(out[column])
    out["TotalCharges"] = _text(out["TotalCharges"]).fillna("").astype(object)
    return out


def consistency_warnings(record: dict) -> list[str]:
    """Combinations real customers never have. Warned about, not blocked."""
    found = []
    internet = record.get("InternetService")
    if internet == "No":
        bad = [c for c in ADDON_COLUMNS if record.get(c) != "No internet service"]
        if bad:
            found.append(f"InternetService is 'No', so {', '.join(bad)} should be "
                         "'No internet service'.")
    elif internet in ("DSL", "Fiber optic"):
        bad = [c for c in ADDON_COLUMNS if record.get(c) == "No internet service"]
        if bad:
            found.append(f"The customer has internet, but {', '.join(bad)} say "
                         "'No internet service'.")
    phone, lines = record.get("PhoneService"), record.get("MultipleLines")
    if phone == "No" and lines != "No phone service":
        found.append("PhoneService is 'No', so MultipleLines should be 'No phone service'.")
    if phone == "Yes" and lines == "No phone service":
        found.append("PhoneService is 'Yes', but MultipleLines says 'No phone service'.")
    try:
        tenure = float(record.get("tenure"))
        monthly = float(record.get("MonthlyCharges"))
        total = float(record.get("TotalCharges"))
        if tenure >= 1 and total + 1e-9 < monthly:
            found.append("TotalCharges is lower than one month's charge for a customer with "
                         "at least one month of tenure.")
    except (TypeError, ValueError):
        pass
    return found


# ---------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------
@dataclass
class Prediction:
    probability: float
    threshold: float
    predicted_churn: bool
    risk_band: str
    segment: str | None
    drivers: pd.DataFrame | None
    base_value: float | None
    model_output: float | None
    units: str | None
    families: dict = field(default_factory=dict)
    recommendations: dict = field(default_factory=dict)


class PredictionService:
    """Loads artifacts once; scores, explains and suggests. Never trains."""

    def __init__(self, estimator, metadata: dict, background: pd.DataFrame | None = None,
                 segment_model=None, rule_config: RuleConfig | None = None,
                 seed: int = 42) -> None:
        self.estimator = estimator
        self.metadata = metadata
        self.background = background
        self.segment_model = segment_model
        self.rule_config = rule_config or RuleConfig()
        self.seed = seed
        self.default_bands = bands_from_metadata(metadata)
        self._explainer = None

    @classmethod
    def from_artifacts(cls, settings: Settings | None = None) -> "PredictionService":
        settings = settings or get_settings()
        final_dir = settings.models_dir / "final"
        model_path, meta_path = final_dir / FINAL_MODEL_FILE, final_dir / METADATA_FILE
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No final model in {final_dir}. Run scripts\\07_finalize_model.py --evaluate-test"
            )
        estimator = joblib.load(model_path)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        background_path = settings.artifacts_dir / "serving" / BACKGROUND_FILE
        background = pd.read_csv(background_path) if background_path.exists() else None
        segment_path = settings.segmentation_dir / "segment_model.joblib"
        segment_model = joblib.load(segment_path) if segment_path.exists() else None
        config_path = settings.artifacts_dir / "recommendations" / "recommendation_config.json"
        rule_config = None
        if config_path.exists():
            raw = json.loads(config_path.read_text(encoding="utf-8")).get("rule_config", {})
            rule_config = RuleConfig(**raw)
        logger.info("PredictionService ready (explanations: %s, segments: %s)",
                    background is not None, segment_model is not None)
        return cls(estimator, metadata, background, segment_model, rule_config,
                   seed=settings.random_seed)

    # ---- properties ---------------------------------------------------
    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"]["value"])

    @property
    def input_columns(self) -> list[str]:
        return list(schema.RAW_FEATURE_COLUMNS)

    @property
    def can_explain(self) -> bool:
        return self.background is not None

    @property
    def explainer(self):
        """SHAP explainer, created on first use (importing shap is slow)."""
        if self._explainer is None:
            if self.background is None:
                raise RuntimeError("Explanations need artifacts\\serving\\background.csv. "
                                   "Run scripts\\11_prepare_serving.py")
            from churn.explain.shap_explainer import ChurnExplainer

            self._explainer = ChurnExplainer(self.estimator, normalize_input(self.background),
                                             seed=self.seed)
        return self._explainer

    def model_card(self) -> dict:
        m = self.metadata
        test = m.get("test_metrics", {})
        ci = test.get("confidence_intervals", {})
        return {
            "model_label": m.get("model_label", m.get("model", "model")),
            "calibration": m.get("calibration", ""),
            "threshold": self.threshold,
            "threshold_policy": m.get("threshold", {}).get("policy", ""),
            "roc_auc": ci.get("roc_auc", {}), "pr_auc": ci.get("pr_auc", {}),
            "brier": ci.get("brier", {}), "at_threshold": test.get("at_threshold", {}),
            "selected_by": m.get("selected_by", ""), "selection_reason": m.get("selection_reason", ""),
            "created_utc": m.get("created_utc", ""),
        }

    def bands_for(self, threshold: float | None = None) -> BandThresholds:
        if threshold is None:
            return self.default_bands
        return adjust_decision(self.default_bands, float(threshold))

    # ---- scoring ------------------------------------------------------
    def check_file(self, df: pd.DataFrame) -> list[str]:
        """File-level problems that stop a whole batch."""
        errors = []
        if df.empty:
            errors.append("The file contains no rows.")
        if len(df) > MAX_BATCH_ROWS:
            errors.append(f"The file has {len(df):,} rows; the limit is {MAX_BATCH_ROWS:,}.")
        missing = [c for c in self.input_columns if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
        return errors

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X[self.input_columns]))[:, 1]

    def predict_batch(self, df: pd.DataFrame, threshold: float | None = None,
                      explain: bool = True, top_actions: int = 3) -> pd.DataFrame:
        """Score every valid row; invalid rows are kept with the reason."""
        errors = self.check_file(df)
        if errors:
            raise BatchInputError(errors)
        data = df.reset_index(drop=True)
        problems = row_problems(data)
        valid = (problems == "").to_numpy()

        out = pd.DataFrame({"row": np.arange(1, len(data) + 1)})
        if schema.ID_COLUMN in data.columns:
            out[schema.ID_COLUMN] = _text(data[schema.ID_COLUMN]).fillna("").astype(object)
        out["status"] = np.where(valid, "scored", "rejected")
        out["problem"] = problems.to_numpy(dtype=object)
        out["churn_probability"] = np.nan
        text_columns = ["predicted_churn", "risk_band", "segment", "top_driver",
                        *[f"action_{j + 1}" for j in range(top_actions)]]
        for column in text_columns:
            out[column] = pd.Series([""] * len(out), dtype=object)
        if not valid.any():
            return out

        idx = np.flatnonzero(valid)
        X = normalize_input(data.loc[valid])
        prob = self.predict_proba(X)
        bands = self.bands_for(threshold)
        band = assign_risk_band(prob, bands)
        out.loc[idx, "churn_probability"] = prob
        out.loc[idx, "predicted_churn"] = np.where(prob >= bands.decision, "Yes", "No")
        out.loc[idx, "risk_band"] = band
        if self.segment_model is not None:
            out.loc[idx, "segment"] = self.segment_model.segment_names(X)

        if explain and self.can_explain:
            families = family_contributions(self.explainer.shap_values(X), self.explainer.groups)
            records = customer_records(X)
            tops, actions = [], [[] for _ in range(top_actions)]
            for i in range(len(X)):
                contrib = families.iloc[i].to_dict()
                top = max(contrib, key=contrib.get)
                tops.append(top if contrib[top] > 0 else "")
                rec = recommend(records[i], float(prob[i]), str(band[i]), contrib,
                                self.rule_config, max_items=top_actions)
                titles = [a["title"] for a in rec["actions"]]
                for j in range(top_actions):
                    actions[j].append(titles[j] if j < len(titles) else "")
            out.loc[idx, "top_driver"] = tops
            for j in range(top_actions):
                out.loc[idx, f"action_{j + 1}"] = actions[j]
        return out

    def predict_one(self, record: dict, threshold: float | None = None, top_n: int = 8,
                    max_actions: int = 3) -> Prediction:
        """Score, explain and suggest actions for ONE customer."""
        missing = [c for c in self.input_columns if c not in record]
        if missing:
            raise BatchInputError([f"Missing fields: {missing}"])
        frame = pd.DataFrame([{c: record[c] for c in self.input_columns}])
        problem = row_problems(frame).iloc[0]
        if problem:
            raise BatchInputError(problem.split("; "))

        X = normalize_input(frame)
        probability = float(self.predict_proba(X)[0])
        bands = self.bands_for(threshold)
        band = str(assign_risk_band([probability], bands)[0])
        segment = self.segment_model.segment_names(X)[0] if self.segment_model is not None else None

        drivers = base = output = units = None
        families: dict = {}
        if self.can_explain:
            from churn.explain.shap_explainer import explain_customer

            explanation = explain_customer(self.explainer, X, top_n=top_n)
            drivers = explanation["contributions"]
            base, output = explanation["base_value"], explanation["model_output"]
            units = explanation["units"]
            families = family_contributions(self.explainer.shap_values(X),
                                            self.explainer.groups).iloc[0].to_dict()
        suggestions = recommend(customer_records(X)[0], probability, band, families or None,
                                self.rule_config, max_items=max_actions)
        return Prediction(probability=probability, threshold=bands.decision,
                          predicted_churn=probability >= bands.decision, risk_band=band,
                          segment=segment, drivers=drivers, base_value=base,
                          model_output=output, units=units, families=families,
                          recommendations=suggestions)