"""Milestone 6B: one-time test evaluation and the saved final model."""
from __future__ import annotations

import json
from importlib import metadata as importlib_metadata
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

from churn.config.settings import BusinessSettings  # noqa: E402
from churn.models.calibration import probability_metrics  # noqa: E402
from churn.models.evaluation import (  # noqa: E402
    CLASS_NAMES,
    classification_report_text,
    compute_metrics,
)
from churn.models.selection import bootstrap_ci  # noqa: E402
from churn.models.threshold import net_benefit  # noqa: E402

FINAL_MODEL_FILE = "churn_pipeline.joblib"
METADATA_FILE = "metadata.json"
_PACKAGES = ["scikit-learn", "imbalanced-learn", "xgboost", "lightgbm", "catboost",
             "pandas", "numpy", "joblib"]


def package_versions() -> dict:
    found = {}
    for name in _PACKAGES:
        try:
            found[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            found[name] = "not installed"
    return found


def base_pipeline(estimator):
    """The fitted feature + model pipeline inside an (optionally) calibrated estimator."""
    if isinstance(estimator, CalibratedClassifierCV):
        return estimator.calibrated_classifiers_[0].estimator
    return estimator


def check_not_finalized(final_dir: Path | str, force: bool) -> int:
    """Return how many test evaluations already happened.

    Raises FileExistsError if a final model exists and force is False, so the
    test split is not reused by accident.
    """
    path = Path(final_dir) / METADATA_FILE
    if not path.exists():
        return 0
    previous = json.loads(path.read_text(encoding="utf-8"))
    count = int(previous.get("test_evaluation_count", 1))
    if not force:
        raise FileExistsError(
            f"A final model already exists ({path}); the test split has been used {count} time(s). "
            "Rerun with --force only for a genuine technical rerun. It will be recorded."
        )
    return count


def metrics_at_threshold(y_true, y_prob, threshold: float, business: BusinessSettings) -> dict:
    """All classification metrics plus business figures at one threshold."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= threshold).astype(int)
    m = compute_metrics(y, pred, p)
    benefit = net_benefit(m["tp"], m["fp"], business)
    m.update({
        "threshold": float(threshold),
        "flagged_pct": 100.0 * float(pred.mean()),
        "net_benefit": benefit,
        "net_benefit_per_1000": 1000.0 * benefit / len(y),
    })
    return m


def evaluate_on_test(estimator, X_test, y_test, thresholds: dict, chosen_policy: str,
                     business: BusinessSettings, n_boot: int = 2000, seed: int = 42) -> dict:
    """Score the final estimator on the test split at every stored threshold."""
    y = np.asarray(y_test).astype(int)
    prob = np.asarray(estimator.predict_proba(X_test))[:, 1]
    chosen = float(thresholds[chosen_policy])
    return {
        "n_rows": int(len(y)),
        "churn_rate": float(y.mean()),
        "probability_metrics": probability_metrics(y, prob),
        "confidence_intervals": {m: bootstrap_ci(y, prob, m, n_boot, seed)
                                 for m in ("roc_auc", "pr_auc", "brier")},
        "policies": {policy: metrics_at_threshold(y, prob, t, business)
                     for policy, t in thresholds.items()},
        "chosen_policy": chosen_policy,
        "chosen_threshold": chosen,
        "classification_report": classification_report_text(y, (prob >= chosen).astype(int)),
        "probabilities": prob,
        "y_true": y,
    }


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Cannot serialise {type(obj)}")


def save_final(estimator, metadata: dict, evaluation: dict,
               final_dir: Path | str, metrics_dir: Path | str) -> list[Path]:
    """Save the final estimator, its metadata and the test results."""
    final_dir, metrics_dir = Path(final_dir), Path(metrics_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model_path = final_dir / FINAL_MODEL_FILE
    joblib.dump(estimator, model_path)
    meta_path = final_dir / METADATA_FILE
    meta_path.write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")

    summary = {k: v for k, v in evaluation.items()
               if k not in ("probabilities", "y_true", "classification_report")}
    metrics_path = metrics_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    report_path = metrics_dir / "test_classification_report.txt"
    report_path.write_text(evaluation["classification_report"], encoding="utf-8")
    probs_path = metrics_dir / "test_probabilities.npz"
    np.savez_compressed(probs_path, y_true=evaluation["y_true"].astype(np.int8),
                        probability=evaluation["probabilities"])
    return [model_path, meta_path, metrics_path, report_path, probs_path]


def _save(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_test_curves(y_true, y_prob, threshold: float, path: Path | str) -> Path:
    """ROC and precision-recall curves with the chosen operating point."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    tn = int(((~pred) & (y == 0)).sum())

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y, p)
    ax_roc.plot(fpr, tpr, label="Final model")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax_roc.scatter([fp / (fp + tn) if fp + tn else 0], [tp / (tp + fn) if tp + fn else 0],
                   color="#C44E52", zorder=3, label=f"Threshold {threshold:.2f}")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate (recall)")
    ax_roc.set_title("ROC curve (test split)")
    ax_roc.legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(y, p)
    ax_pr.plot(recall, precision, label="Final model")
    ax_pr.axhline(float(y.mean()), color="k", linestyle="--", linewidth=0.8,
                  label="Churn rate (random model)")
    ax_pr.scatter([tp / (tp + fn) if tp + fn else 0], [tp / (tp + fp) if tp + fp else 0],
                  color="#C44E52", zorder=3, label=f"Threshold {threshold:.2f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-recall curve (test split)")
    ax_pr.legend(loc="upper right")
    return _save(fig, path)


def plot_test_calibration(y_true, y_prob, path: Path | str, n_bins: int = 10) -> Path:
    """Calibration curve plus the distribution of predicted probabilities."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    observed, predicted = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    fig, (ax_cal, ax_hist) = plt.subplots(2, 1, figsize=(6.5, 7.5),
                                          gridspec_kw={"height_ratios": [3, 1]})
    ax_cal.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    ax_cal.plot(predicted, observed, marker="o", label="Final model")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_ylabel("Observed churn rate")
    ax_cal.set_title("Calibration on the test split")
    ax_cal.legend(loc="upper left")
    ax_hist.hist(p, bins=20, range=(0, 1), color="#4C72B0")
    ax_hist.set_xlabel("Predicted churn probability")
    ax_hist.set_ylabel("Customers")
    return _save(fig, path)


def plot_test_confusion(y_true, y_prob, threshold: float, path: Path | str) -> Path:
    """Confusion matrix at the chosen threshold."""
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center",
                color="white" if value > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Test split, threshold {threshold:.2f}")
    return _save(fig, path)


def format_final_report(metadata: dict, evaluation: dict) -> str:
    """Plain-text summary of the final model and its one-time test evaluation."""
    ci = evaluation["confidence_intervals"]
    chosen = evaluation["policies"][evaluation["chosen_policy"]]
    val = metadata["validation_metrics"]
    lines = [
        f"FINAL MODEL: {metadata['model_label']} ({metadata['calibration']})",
        f"  Selected by: {metadata['selected_by']}. {metadata['selection_reason']}",
        f"  Threshold: {metadata['threshold']['value']:.2f} "
        f"(policy '{metadata['threshold']['policy']}', chosen on the validation split)",
        f"  Test evaluation count: {metadata['test_evaluation_count']}",
        "",
        f"TEST SPLIT ({evaluation['n_rows']} customers, churn rate {evaluation['churn_rate']:.2%}) "
        "- threshold-free metrics with 95% bootstrap intervals",
    ]
    for name in ("roc_auc", "pr_auc", "brier"):
        c = ci[name]
        lines.append(f"  {name:<8} {c['value']:.4f}  (95% CI {c['ci_low']:.4f} to {c['ci_high']:.4f})")
    pm = evaluation["probability_metrics"]
    lines += [
        f"  log_loss {pm['log_loss']:.4f}   ece {pm['ece']:.4f}   "
        f"mean predicted {pm['mean_predicted']:.3f} vs observed {pm['observed_rate']:.3f}",
        "",
        "VALIDATION vs TEST (same model, same threshold)",
        f"  PR-AUC   validation {val['pr_auc']:.4f}   test {ci['pr_auc']['value']:.4f}",
        f"  ROC-AUC  validation {val['roc_auc']:.4f}   test {ci['roc_auc']['value']:.4f}",
        f"  Brier    validation {val['brier']:.4f}   test {ci['brier']['value']:.4f}",
        f"  Recall   validation {val['at_threshold']['recall']:.3f}    test {chosen['recall']:.3f}",
        f"  Precision validation {val['at_threshold']['precision']:.3f}   test {chosen['precision']:.3f}",
        "",
        "TEST METRICS AT EACH STORED THRESHOLD",
        "  policy            threshold  precision  recall   f1     specificity  bal_acc  mcc    flagged%  net/1000",
    ]
    for policy, m in evaluation["policies"].items():
        lines.append(
            f"  {policy:<17} {m['threshold']:<10.2f} {m['precision']:<10.3f} {m['recall']:<8.3f} "
            f"{m['f1']:<6.3f} {m['specificity']:<12.3f} {m['balanced_accuracy']:<8.3f} "
            f"{m['mcc']:<6.3f} {m['flagged_pct']:<9.1f} {m['net_benefit_per_1000']:.1f}"
        )
    lines += [
        "",
        "CLASSIFICATION REPORT (chosen threshold)",
        evaluation["classification_report"],
        "Net benefit uses PLACEHOLDER business costs and is illustrative only.",
    ]
    return "\n".join(lines)