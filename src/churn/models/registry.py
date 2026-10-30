"""Registry of the 13 benchmarked algorithms.

Every model is created through build_estimator(), so seeds, thread counts
and class-weight handling are decided in exactly one place. These are
reasonable baseline settings, not tuned ones. Tuning happens in Milestone 5.

All models run single-threaded (n_jobs=1) so training times are comparable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# factory(seed, weighted, pos_weight) -> unfitted estimator
#   weighted   : True when the "class_weight" strategy is active
#   pos_weight : (number of retained) / (number of churned) in the training data
Factory = Callable[[int, bool, float], BaseEstimator]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str
    factory: Factory
    supports_class_weight: bool
    class_weight_note: str


def _logistic_regression(seed: int, weighted: bool, pos_weight: float) -> BaseEstimator:
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced" if weighted else None,
        random_state=seed,
    )


def _knn(seed: int, weighted: bool, pos_weight: float) -> BaseEstimator:
    return KNeighborsClassifier(n_neighbors=15)


def _decision_tree(seed: int, weighted: bool, pos_weight: float) -> BaseEstimator:
    return DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced" if weighted else None,
        random_state=seed,
    )


def _naive_bayes(seed: int, weighted: bool, pos_weight: float) -> BaseEstimator:
    return GaussianNB()


def _svm(seed: int, weighted: bool, pos_weight: float) -> BaseEstimator:
    # probability=False: much faster. Scores come from decision_function.
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=False,
        class_weight="balanced" if weighted else None,
        cache_size=500,
        random_state=seed,
    )


def _random_forest(seed: int, weighted: bool, pos_weight: float) -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced" if weighted else None,
        n_jobs=1,
        random_state=seed,
    )


def _extra_trees(seed: int, weighted: bool, pos_weight: float) -> BaseEstimator:
    return ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced" if weighted else None,
        n_jobs=1,
        random_state=seed,
    )


def _adaboost(seed: int, weighted: bool, pos_weight: float) -> BaseEstimator:
    return AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=seed)


def _gradient_boosting(seed: int, weighted: bool, pos_weight: float) -> BaseEstimator:
    return GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=seed
    )


def _hist_gradient_boosting(seed: int, weighted: bool, pos_weight: float) -> BaseEstimator:
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        class_weight="balanced" if weighted else None,
        random_state=seed,
    )


def _xgboost(seed: int, weighted: bool, pos_weight: float) -> BaseEstimator:
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight if weighted else 1.0,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
    )


def _lightgbm(seed: int, weighted: bool, pos_weight: float) -> BaseEstimator:
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        class_weight="balanced" if weighted else None,
        n_jobs=1,
        random_state=seed,
        verbose=-1,
    )


def _catboost(seed: int, weighted: bool, pos_weight: float) -> BaseEstimator:
    return CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        auto_class_weights="Balanced" if weighted else None,
        thread_count=1,
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
    )


_NO_WEIGHT_NOTE = "{name} has no class_weight parameter, so the class_weight strategy does not apply."

_SPECS = [
    ModelSpec("logistic_regression", "Logistic Regression", "classical",
              _logistic_regression, True, "class_weight='balanced'"),
    ModelSpec("knn", "KNN", "classical", _knn, False,
              "KNeighborsClassifier cannot re-weight votes by class, so the class_weight strategy does not apply."),
    ModelSpec("decision_tree", "Decision Tree", "classical",
              _decision_tree, True, "class_weight='balanced'"),
    ModelSpec("naive_bayes", "Gaussian NB", "classical", _naive_bayes, False,
              "GaussianNB has only fixed class priors, no class_weight parameter, so the class_weight strategy does not apply."),
    ModelSpec("svm", "SVM (RBF)", "classical", _svm, True, "class_weight='balanced'"),
    ModelSpec("random_forest", "Random Forest", "ensemble",
              _random_forest, True, "class_weight='balanced'"),
    ModelSpec("extra_trees", "Extra Trees", "ensemble",
              _extra_trees, True, "class_weight='balanced'"),
    ModelSpec("adaboost", "AdaBoost", "ensemble", _adaboost, False,
              _NO_WEIGHT_NOTE.format(name="AdaBoostClassifier")),
    ModelSpec("gradient_boosting", "Gradient Boosting", "ensemble", _gradient_boosting, False,
              _NO_WEIGHT_NOTE.format(name="GradientBoostingClassifier")),
    ModelSpec("hist_gradient_boosting", "HistGradientBoosting", "ensemble",
              _hist_gradient_boosting, True, "class_weight='balanced'"),
    ModelSpec("xgboost", "XGBoost", "boosting",
              _xgboost, True, "scale_pos_weight = negatives / positives"),
    ModelSpec("lightgbm", "LightGBM", "boosting",
              _lightgbm, True, "class_weight='balanced'"),
    ModelSpec("catboost", "CatBoost", "boosting",
              _catboost, True, "auto_class_weights='Balanced'"),
]

REGISTRY: dict[str, ModelSpec] = {spec.key: spec for spec in _SPECS}


def list_model_keys() -> list[str]:
    """Model keys in benchmark order."""
    return list(REGISTRY)


def get_spec(key: str) -> ModelSpec:
    if key not in REGISTRY:
        raise KeyError(f"Unknown model '{key}'. Available: {list_model_keys()}")
    return REGISTRY[key]


def build_estimator(
    key: str, seed: int, weighted: bool = False, pos_weight: float = 1.0
) -> BaseEstimator:
    """Create a fresh, unfitted estimator."""
    return get_spec(key).factory(seed, weighted, pos_weight)