"""Milestone 6A: calibrate the tuned models and analyse thresholds on VALIDATION data.

Every estimator is fitted on the training split only. The validation split
is used to compare calibration methods and choose thresholds. The test
split is not used here; it is opened once, in Milestone 6B.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, learning_curve

from churn.config.settings import BusinessSettings
from churn.models.calibration import CALIBRATION_METHODS, build_calibrated, probability_metrics
from churn.models.threshold import breakeven_probability, select_thresholds, sweep_thresholds
from churn.utils.logger import get_logger

logger = get_logger(__name__)

PROBABILITY_COLUMNS = ["roc_auc", "pr_auc", "brier", "log_loss", "ece",
                       "mean_predicted", "observed_rate"]


@dataclass
class CandidateResult:
    model: str
    calibration: str
    status: str = "ok"
    reason: str = ""
    fit_seconds: float = float("nan")
    metrics: dict = field(default_factory=dict)
    val_prob: np.ndarray | None = None
    estimator: object | None = None

    @property
    def key(self) -> str:
        return f"{self.model}__{self.calibration}"


def load_tuned_pipelines(directory: Path | str) -> dict:
    """Load every tuned pipeline saved by Milestone 5, keyed by model name."""
    directory = Path(directory)
    files = sorted(directory.glob("*.joblib")) if directory.exists() else []
    if not files:
        raise FileNotFoundError(
            f"No tuned models found in {directory}. Run: python scripts\\05_tune_models.py"
        )
    return {path.stem: joblib.load(path) for path in files}


def evaluate_candidates(
    tuned: dict,
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame,
    y_val,
    methods=CALIBRATION_METHODS,
    seed: int = 42,
    n_splits: int = 5,
    n_jobs: int = 1,
) -> list[CandidateResult]:
    """Fit every model x calibration method on train, score on validation."""
    X_train = X_train.reset_index(drop=True)
    y_tr = np.asarray(y_train).astype(int)
    y_v = np.asarray(y_val).astype(int)
    methods = list(methods)
    total = len(tuned) * len(methods)
    results: list[CandidateResult] = []
    position = 0
    for model, pipeline in tuned.items():
        for method in methods:
            position += 1
            try:
                estimator = build_calibrated(pipeline, method, seed=seed,
                                             n_splits=n_splits, n_jobs=n_jobs)
                started = perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    warnings.simplefilter("ignore", category=FutureWarning)
                    warnings.simplefilter("ignore", category=DeprecationWarning)
                    estimator.fit(X_train, y_tr)
                fit_seconds = perf_counter() - started
                prob = np.asarray(estimator.predict_proba(X_val))[:, 1]
                metrics = probability_metrics(y_v, prob)
                results.append(CandidateResult(model, method, fit_seconds=fit_seconds,
                                               metrics=metrics, val_prob=prob,
                                               estimator=estimator))
                logger.info("[%d/%d] %s + %s: validation PR-AUC %.4f, Brier %.4f (fit %.1fs)",
                            position, total, model, method, metrics["pr_auc"],
                            metrics["brier"], fit_seconds)
            except Exception as exc:  # noqa: BLE001 - record and continue
                reason = f"{type(exc).__name__}: {str(exc)[:300]}"
                results.append(CandidateResult(model, method, status="failed", reason=reason))
                logger.error("[%d/%d] %s + %s FAILED: %s", position, total, model, method, reason)
    return results


def comparison_table(results: list[CandidateResult]) -> pd.DataFrame:
    """Validation metrics, best ranking first (ties broken by lower Brier)."""
    rows = [{"model": r.model, "calibration": r.calibration, **r.metrics,
             "fit_seconds": r.fit_seconds}
            for r in results if r.status == "ok"]
    table = pd.DataFrame(rows, columns=["model", "calibration", *PROBABILITY_COLUMNS,
                                        "fit_seconds"])
    return table.sort_values(["pr_auc", "brier"], ascending=[False, True]).reset_index(drop=True)


def best_calibration_per_model(comparison: pd.DataFrame) -> pd.DataFrame:
    """For each model, the calibration method with the lowest validation Brier score."""
    return (comparison.sort_values("brier")
            .drop_duplicates("model")
            .sort_values("pr_auc", ascending=False)
            .reset_index(drop=True))


def threshold_tables(results, y_val, business: BusinessSettings,
                     recall_target: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(full threshold sweep, chosen thresholds per policy) for every candidate."""
    sweeps, choices = [], []
    for r in results:
        if r.status != "ok":
            continue
        sweep = sweep_thresholds(y_val, r.val_prob, business)
        sweep.insert(0, "calibration", r.calibration)
        sweep.insert(0, "model", r.model)
        sweeps.append(sweep)
        choices.append(select_thresholds(sweep, recall_target))
    if not sweeps:
        return pd.DataFrame(), pd.DataFrame()
    return (pd.concat(sweeps, ignore_index=True),
            pd.concat(choices, ignore_index=True))


def compute_learning_curve(pipeline, X: pd.DataFrame, y, seed: int = 42, n_splits: int = 5,
                           n_jobs: int = 1, train_sizes=(0.2, 0.4, 0.6, 0.8, 1.0)) -> dict:
    """Training and CV PR-AUC as the number of training rows grows."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        sizes, train_scores, cv_scores = learning_curve(
            clone(pipeline), X.reset_index(drop=True), np.asarray(y).astype(int),
            train_sizes=np.asarray(train_sizes), cv=cv, scoring="average_precision",
            n_jobs=n_jobs, shuffle=True, random_state=seed,
        )
    return {
        "sizes": sizes,
        "train_mean": train_scores.mean(axis=1), "train_std": train_scores.std(axis=1),
        "cv_mean": cv_scores.mean(axis=1), "cv_std": cv_scores.std(axis=1),
    }


def learning_curve_table(curves: dict[str, dict]) -> pd.DataFrame:
    """Long-format table of all learning curves."""
    rows = []
    for label, c in curves.items():
        for i, size in enumerate(c["sizes"]):
            rows.append({"model": label, "train_rows": int(size),
                         "train_pr_auc": float(c["train_mean"][i]),
                         "train_std": float(c["train_std"][i]),
                         "cv_pr_auc": float(c["cv_mean"][i]),
                         "cv_std": float(c["cv_std"][i])})
    return pd.DataFrame(rows)


def save_candidates(results, comparison, sweep, choices, y_val,
                    metrics_dir: Path | str, models_dir: Path | str,
                    meta: dict | None = None) -> list[Path]:
    """Write candidate estimators, tables, validation probabilities and metadata."""
    metrics_dir, models_dir = Path(metrics_dir), Path(models_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    arrays = {"y_val": np.asarray(y_val).astype(np.int8)}

    for r in results:
        if r.status != "ok":
            continue
        path = models_dir / f"{r.key}.joblib"
        joblib.dump(r.estimator, path)
        arrays[r.key] = r.val_prob
        written.append(path)

    for name, table in {"validation_comparison": comparison,
                        "threshold_sweep": sweep,
                        "threshold_choices": choices}.items():
        path = metrics_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        written.append(path)

    prob_path = metrics_dir / "validation_probabilities.npz"
    np.savez_compressed(prob_path, **arrays)
    written.append(prob_path)

    if meta is not None:
        meta_path = metrics_dir / "calibration_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        written.append(meta_path)
    return written


def format_candidates_report(comparison: pd.DataFrame, choices: pd.DataFrame, results,
                             business: BusinessSettings, recall_target: float) -> str:
    """Plain-text report for the console and calibration_report.txt."""
    lines: list[str] = []
    if comparison.empty:
        lines.append("No successful candidates.")
    else:
        view = comparison[["model", "calibration", "roc_auc", "pr_auc", "brier",
                           "log_loss", "ece", "fit_seconds"]].round(4)
        best = best_calibration_per_model(comparison)
        picked = choices.merge(best[["model", "calibration"]], on=["model", "calibration"])
        tview = picked[["model", "calibration", "policy", "threshold", "precision", "recall",
                        "f1", "flagged_pct", "net_benefit_per_1000"]].round(3)
        lines += [
            "VALIDATION COMPARISON (fitted on the training split, scored on the validation split)",
            view.to_string(index=False),
            "",
            "roc_auc / pr_auc: ranking quality (higher is better).",
            "brier / log_loss / ece: probability quality (lower is better).",
            "",
            "BUSINESS ASSUMPTIONS (PLACEHOLDERS - replace them in config.yaml when real figures are known)",
            f"  customer_value = {business.customer_value:g} {business.currency}, "
            f"offer_cost = {business.offer_cost:g} {business.currency}, "
            f"save_rate = {business.save_rate:g}",
            "  net_benefit = TP x save_rate x customer_value - (TP + FP) x offer_cost",
            "  break-even probability = offer_cost / (save_rate x customer_value) = "
            f"{breakeven_probability(business):.3f}",
            "",
            f"THRESHOLD CHOICES (lowest-Brier calibration per model; recall target {recall_target:.2f})",
            tview.to_string(index=False),
        ]

    failed = [r for r in results if r.status != "ok"]
    if failed:
        lines += ["", "FAILED CANDIDATES"]
        lines += [f"  {r.model} + {r.calibration}: {r.reason}" for r in failed]

    lines += [
        "",
        "Note: thresholds are chosen and scored on the same validation rows, so these numbers",
        "are slightly optimistic. The test split (Milestone 6B) gives the final, honest estimate.",
    ]
    return "\n".join(lines)