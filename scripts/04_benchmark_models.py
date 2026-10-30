"""Benchmark all models and imbalance strategies on the TRAINING split.

The validation and test splits are not touched here.
Examples (CMD):
    python scripts\\04_benchmark_models.py
    python scripts\\04_benchmark_models.py --models logistic_regression,xgboost --strategies none,class_weight
"""
from __future__ import annotations

import argparse
import sys

from churn.config.settings import get_settings
from churn.data.fingerprint import short_fingerprint
from churn.features.splitting import load_split, split_xy
from churn.models.benchmark import benchmark_models, format_report, save_benchmark
from churn.models.benchmark_plots import make_all_plots
from churn.models.imbalance import STRATEGIES
from churn.models.registry import list_model_keys
from churn.utils.logger import get_logger

logger = get_logger("benchmark_models")


def _parse_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", help="Comma-separated model keys (default: all 13).")
    parser.add_argument("--strategies", help="Comma-separated strategies (default: all 5).")
    parser.add_argument("--folds", type=int, help="CV folds (default: cv.n_splits in config.yaml).")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel runs. 1 gives the cleanest timings (default).")
    args = parser.parse_args()

    models = _parse_list(args.models)
    strategies = _parse_list(args.strategies)
    for name in models or []:
        if name not in list_model_keys():
            logger.error("Unknown model '%s'. Available: %s", name, list_model_keys())
            return 1
    for name in strategies or []:
        if name not in STRATEGIES:
            logger.error("Unknown strategy '%s'. Available: %s", name, list(STRATEGIES))
            return 1

    settings = get_settings()
    settings.ensure_directories()
    train = load_split("train", settings=settings)
    X_train, y_train = split_xy(train, settings)
    logger.info("Training split: %d rows, churn rate %.2f%%",
                len(X_train), y_train.mean() * 100)

    result = benchmark_models(
        X_train, y_train,
        model_keys=models, strategies=strategies,
        n_splits=args.folds or settings.cv_splits,
        seed=settings.random_seed, n_jobs=args.n_jobs,
        extra_meta={"source_fingerprint": short_fingerprint(settings.raw_data_path),
                    "split": "train"},
    )

    out_dir = settings.metrics_dir / "benchmark"
    paths = save_benchmark(result, out_dir)
    plot_files = make_all_plots(result, settings.plots_dir / "benchmark")

    report = format_report(result)
    report_path = out_dir / "benchmark_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print("\n" + report)

    for path in [*paths.values(), report_path, *plot_files]:
        logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())