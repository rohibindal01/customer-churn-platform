"""Keep the Docker setup honest: lean runtime, pinned versions, nothing private in the image."""
import re
import subprocess
import sys

import pytest

from churn.config.settings import _resolve_root, get_settings

ROOT = get_settings().root
TRAINING_ONLY = ("xgboost", "lightgbm", "catboost", "mlflow", "seaborn", "kaggle")


def _lines(name: str) -> list[str]:
    text = (ROOT / name).read_text(encoding="utf-8")
    return [line.split("#")[0].strip() for line in text.splitlines()
            if line.split("#")[0].strip()]


def test_serving_path_does_not_import_training_libraries():
    code = (
        "import sys\n"
        "import churn.serving.predictor, churn.explain.shap_explainer\n"
        "import churn.analytics.segmentation, churn.models.registry, churn.models.imbalance\n"
        f"print(','.join(m for m in {TRAINING_ONLY!r} if m in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"Imported training-only libraries: {result.stdout}"


def test_runtime_requirements_are_pinned_by_the_lock_file():
    lock = ROOT / "requirements-lock.txt"
    if not lock.exists():
        pytest.skip("requirements-lock.txt missing: run pip freeze (Milestone 11, Step 2)")
    pinned = {line.split("==")[0].strip().lower().replace("_", "-")
              for line in lock.read_text(encoding="utf-8").splitlines() if "==" in line}
    for requirement in _lines("requirements-app.txt"):
        name = re.split(r"[<>=!~\[ ]", requirement)[0].lower().replace("_", "-")
        assert name in pinned, f"{name} is not pinned in requirements-lock.txt"
        assert name not in TRAINING_ONLY, f"{name} is training-only"


def test_dockerignore_is_an_allowlist():
    lines = _lines(".dockerignore")
    assert lines[0] == "*"
    for needed in ("!src", "!app", "!artifacts", "!config.yaml", "!requirements-app.txt",
                   "!requirements-lock.txt"):
        assert needed in lines, needed
    for private in ("!data", "!.env", "!.venv", "!.git", "!mlflow.db", "!logs"):
        assert private not in lines, private


def test_dockerfile_matches_the_training_environment():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert f"FROM python:{python}-slim" in text, f"Image Python should be {python}"
    for expected in ("CHURN_PROJECT_ROOT=/app", "-c requirements-lock.txt", "USER appuser",
                     "EXPOSE 8501", "HEALTHCHECK", "/_stcore/health"):
        assert expected in text, expected


def test_compose_maps_the_port_and_keeps_logs():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert '"8501:8501"' in text
    assert "churn-logs:/app/logs" in text
    assert "healthcheck" in text


def test_project_root_can_be_set_by_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CHURN_PROJECT_ROOT", str(tmp_path))
    assert _resolve_root() == tmp_path.resolve()