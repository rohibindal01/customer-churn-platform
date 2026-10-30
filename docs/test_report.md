# Test report

Generated 2026-10-30T22:57:48+00:00 from an actual test run started 2026-10-31T04:25:05.739713+05:30.

Command: `python -m pytest tests --cov --cov-report=term-missing --cov-report=xml:artifacts\tests\coverage.xml --junitxml=artifacts\tests\junit.xml`

## Summary

| Tests | Passed | Failed | Errors | Skipped | Duration |
|---|---|---|---|---|---|
| 283 | 283 | 0 | 0 | 0 | 134 s |

Skipped tests are not passed tests: they did not run (see the reasons below).

## Results by test file

| File | Tests | Passed | Failed | Skipped | Seconds |
|---|---|---|---|---|---|
| `tests/integration/test_app_pages.py` | 3 | 3 | 0 | 0 | 13.0 |
| `tests/integration/test_app_smoke.py` | 11 | 11 | 0 | 0 | 13.7 |
| `tests/integration/test_batch_page.py` | 4 | 4 | 0 | 0 | 4.2 |
| `tests/integration/test_features_real_data.py` | 2 | 2 | 0 | 0 | 0.3 |
| `tests/integration/test_final_artifacts.py` | 4 | 4 | 0 | 0 | 0.1 |
| `tests/integration/test_raw_data_file.py` | 2 | 2 | 0 | 0 | 0.1 |
| `tests/integration/test_reproducibility.py` | 3 | 3 | 0 | 0 | 0.3 |
| `tests/integration/test_scripts_cli.py` | 5 | 5 | 0 | 0 | 15.5 |
| `tests/unit/test_app_charts.py` | 7 | 7 | 0 | 0 | 0.2 |
| `tests/unit/test_benchmark.py` | 7 | 7 | 0 | 0 | 7.0 |
| `tests/unit/test_calibration.py` | 7 | 7 | 0 | 0 | 0.9 |
| `tests/unit/test_candidates.py` | 7 | 7 | 0 | 0 | 8.5 |
| `tests/unit/test_cleaning.py` | 5 | 5 | 0 | 0 | 0.1 |
| `tests/unit/test_docker_files.py` | 6 | 6 | 0 | 0 | 3.4 |
| `tests/unit/test_eda.py` | 8 | 8 | 0 | 0 | 3.8 |
| `tests/unit/test_engineering.py` | 8 | 8 | 0 | 0 | 0.1 |
| `tests/unit/test_evaluation.py` | 5 | 5 | 0 | 0 | 0.2 |
| `tests/unit/test_explain.py` | 11 | 11 | 0 | 0 | 4.2 |
| `tests/unit/test_final.py` | 6 | 6 | 0 | 0 | 4.1 |
| `tests/unit/test_imbalance_and_pipeline.py` | 36 | 36 | 0 | 0 | 7.6 |
| `tests/unit/test_logging.py` | 1 | 1 | 0 | 0 | 0.0 |
| `tests/unit/test_predictor.py` | 12 | 12 | 0 | 0 | 3.7 |
| `tests/unit/test_preprocessing.py` | 7 | 7 | 0 | 0 | 0.9 |
| `tests/unit/test_registry.py` | 17 | 17 | 0 | 0 | 0.1 |
| `tests/unit/test_risk_and_recommendations.py` | 11 | 11 | 0 | 0 | 0.4 |
| `tests/unit/test_schema_and_fingerprint.py` | 5 | 5 | 0 | 0 | 0.0 |
| `tests/unit/test_segmentation.py` | 9 | 9 | 0 | 0 | 3.4 |
| `tests/unit/test_selection.py` | 9 | 9 | 0 | 0 | 11.6 |
| `tests/unit/test_serialization.py` | 14 | 14 | 0 | 0 | 9.9 |
| `tests/unit/test_settings_and_logging.py` | 4 | 4 | 0 | 0 | 0.0 |
| `tests/unit/test_splitting.py` | 6 | 6 | 0 | 0 | 0.2 |
| `tests/unit/test_threshold.py` | 9 | 9 | 0 | 0 | 0.1 |
| `tests/unit/test_tracking.py` | 2 | 2 | 0 | 0 | 8.4 |
| `tests/unit/test_tuning.py` | 17 | 17 | 0 | 0 | 7.5 |
| `tests/unit/test_validator.py` | 13 | 13 | 0 | 0 | 0.4 |

## Coverage

Overall line coverage **94.1%**, branch coverage **72.5%** (source: `src/churn` and `app`).

Least covered first. Coverage shows what did not run; it does not measure test quality.

| File | Lines | Covered | Line coverage | Branch coverage |
|---|---|---|---|---|
| `models/final.py` | 158 | 127 | 80.4% | 33.3% |
| `components/ui.py` | 64 | 54 | 84.4% | 60.0% |
| `pages/3_Batch_Prediction.py` | 96 | 81 | 84.4% | 50.0% |
| `models/benchmark.py` | 143 | 122 | 85.3% | 57.1% |
| `components/loaders.py` | 47 | 42 | 89.4% | 62.5% |
| `config/settings.py` | 128 | 116 | 90.6% | 71.4% |
| `utils/tracking.py` | 32 | 29 | 90.6% | 100.0% |
| `explain/shap_explainer.py` | 157 | 143 | 91.1% | 73.1% |
| `main.py` | 36 | 33 | 91.7% | 50.0% |
| `models/candidates.py` | 138 | 127 | 92.0% | 64.3% |
| `pages/2_Churn_Prediction.py` | 101 | 93 | 92.1% | 61.1% |
| `pages/7_Explainable_AI.py` | 105 | 97 | 92.4% | 50.0% |
| `pages/1_Executive_Dashboard.py` | 75 | 70 | 93.3% | 50.0% |
| `models/selection.py` | 91 | 85 | 93.4% | 90.0% |
| `data/eda.py` | 257 | 241 | 93.8% | 69.0% |
| `data/loader.py` | 19 | 18 | 94.7% | 50.0% |
| `models/tuning.py` | 192 | 183 | 95.3% | 67.7% |
| `models/evaluation.py` | 22 | 21 | 95.5% | 75.0% |
| `features/splitting.py` | 51 | 49 | 96.1% | 75.0% |
| `serving/predictor.py` | 231 | 222 | 96.1% | 83.3% |
| `pages/5_Segmentation.py` | 112 | 108 | 96.4% | 55.6% |
| `pages/8_Retention_Recommendations.py` | 84 | 81 | 96.4% | 50.0% |
| `data/validator.py` | 121 | 117 | 96.7% | 90.0% |
| `pages/4_Risk_Analysis.py` | 72 | 70 | 97.2% | 50.0% |
| `analytics/segmentation.py` | 148 | 144 | 97.3% | 81.2% |
| `models/calibration_plots.py` | 80 | 78 | 97.5% | 83.3% |
| `models/pipeline.py` | 40 | 39 | 97.5% | 90.0% |
| `pages/6_Model_Benchmarking.py` | 163 | 159 | 97.5% | 58.3% |
| `models/benchmark_plots.py` | 98 | 96 | 98.0% | 83.3% |
| `analytics/recommendations.py` | 68 | 67 | 98.5% | 91.7% |
| `__init__.py` | 1 | 1 | 100.0% | 100.0% |
| `analytics/plots.py` | 93 | 93 | 100.0% | 90.0% |
| `analytics/risk.py` | 40 | 40 | 100.0% | 83.3% |
| `components/charts.py` | 84 | 84 | 100.0% | 100.0% |
| `data/fingerprint.py` | 11 | 11 | 100.0% | 100.0% |
| `data/schema.py` | 13 | 13 | 100.0% | 100.0% |
| `explain/permutation.py` | 13 | 13 | 100.0% | 100.0% |
| `explain/plots.py` | 56 | 56 | 100.0% | 75.0% |
| `features/cleaning.py` | 25 | 25 | 100.0% | 87.5% |
| `features/engineering.py` | 27 | 27 | 100.0% | 100.0% |
| `features/preprocessing.py` | 21 | 21 | 100.0% | 100.0% |
| `models/calibration.py` | 30 | 30 | 100.0% | 100.0% |
| `models/imbalance.py` | 21 | 21 | 100.0% | 100.0% |
| `models/registry.py` | 59 | 59 | 100.0% | 100.0% |
| `models/threshold.py` | 44 | 44 | 100.0% | 100.0% |
| `utils/logger.py` | 31 | 31 | 100.0% | 100.0% |

## Environment

- Python 3.12.0 on Windows-11-10.0.26200-SP0
- pytest 9.2.0
- pytest-cov 7.1.0
- scikit-learn 1.9.1
- imbalanced-learn 0.14.2
- xgboost 3.4.1
- lightgbm 4.7.0
- catboost 1.2.10
- shap 0.54.1
- pandas 3.1.0
- numpy 2.5.4
- streamlit 1.66.1
- mlflow 3.17.2
