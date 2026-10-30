"""Verify the interpreter and the key libraries are importable."""
from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = [
    "numpy", "pandas", "scipy", "sklearn", "imblearn", "xgboost",
    "lightgbm", "catboost", "shap", "joblib", "yaml", "matplotlib",
    "seaborn", "plotly", "streamlit", "mlflow", "churn",
]
OPTIONAL = ["interpret"]


def check(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
        return True, str(getattr(module, "__version__", "unknown"))
    except Exception as exc:  # noqa: BLE001 - we want to report any failure
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    print(f"Python   : {sys.version.split()[0]} ({platform.architecture()[0]})")
    print(f"Platform : {platform.platform()}")
    print(f"Prefix   : {sys.prefix}")
    print("-" * 60)

    failures = 0
    for name in REQUIRED:
        ok, info = check(name)
        failures += 0 if ok else 1
        print(f"[{'OK' if ok else 'FAIL':4}] {name:<12} {info}")

    for name in OPTIONAL:
        ok, info = check(name)
        print(f"[{'OK' if ok else 'SKIP':4}] {name:<12} {info} (optional)")

    print("-" * 60)
    print("All required packages import correctly." if failures == 0
          else f"{failures} required package(s) failed to import.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())