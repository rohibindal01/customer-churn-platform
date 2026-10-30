"""Command-line behaviour: environment check, argument validation, real app start-up."""
import socket
import subprocess
import sys
import time
import urllib.request

from churn.config.settings import get_settings

ROOT = get_settings().root


def _run(*args, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout)


def test_environment_check_passes():
    result = _run("scripts/00_check_environment.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All required packages import correctly." in result.stdout


def test_benchmark_rejects_an_unknown_model():
    assert _run("scripts/04_benchmark_models.py", "--models", "not_a_model").returncode == 1


def test_tuning_rejects_a_model_without_search_space():
    assert _run("scripts/05_tune_models.py", "--models", "knn").returncode == 1


def test_finalize_needs_model_and_calibration_together():
    assert _run("scripts/07_finalize_model.py", "--model", "logistic_regression").returncode == 1


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_streamlit_server_starts_and_reports_healthy():
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/main.py",
         "--server.headless", "true", "--server.address", "127.0.0.1",
         "--server.port", str(port), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    body = ""
    try:
        deadline = time.time() + 90
        while time.time() < deadline and process.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health",
                                            timeout=2) as response:
                    body = response.read().decode("utf-8").strip()
                if body == "ok":
                    break
            except OSError:
                time.sleep(1)
        assert body == "ok", "Streamlit did not report healthy within 90 seconds"
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()