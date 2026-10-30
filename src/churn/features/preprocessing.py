"""The shared feature pipeline: clean -> engineer -> encode and scale.

Later milestones put a model after this pipeline, so training and
inference always apply exactly the same steps.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn.data import schema
from churn.features import engineering
from churn.features.cleaning import clean_features
from churn.features.engineering import engineer_features

NUMERIC_FEATURES = [*schema.NUMERIC_COLUMNS, *engineering.ENGINEERED_NUMERIC]
FLAG_FEATURES = [schema.SENIOR_COLUMN, *engineering.ENGINEERED_FLAGS]
CATEGORICAL_FEATURES = [*schema.CATEGORICAL_COLUMNS, *engineering.ENGINEERED_CATEGORICAL]


def build_preprocessor() -> ColumnTransformer:
    """Impute and scale numbers, pass 0/1 flags, one-hot encode categories."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    flags = SimpleImputer(strategy="most_frequent")
    categorical = OneHotEncoder(
        handle_unknown="ignore", drop="if_binary", sparse_output=False
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("flags", flags, FLAG_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline() -> Pipeline:
    """Full feature pipeline. Fit it on training data only."""
    return Pipeline(
        [
            ("clean", FunctionTransformer(clean_features, validate=False)),
            ("engineer", FunctionTransformer(engineer_features, validate=False)),
            ("preprocess", build_preprocessor()),
        ]
    )


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Output column names of a fitted feature pipeline."""
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())