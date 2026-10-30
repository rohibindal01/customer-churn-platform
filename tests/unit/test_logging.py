"""The shared logger writes timestamped lines to logs\\churn.log."""
import logging
import uuid

from churn.config.settings import get_settings
from churn.utils.logger import get_logger


def test_logger_writes_to_the_log_file():
    marker = f"test-marker-{uuid.uuid4()}"
    get_logger("test_logging").info(marker)
    for handler in logging.getLogger("churn").handlers:
        handler.flush()
    text = get_settings().log_path.read_text(encoding="utf-8")
    assert marker in text
    line = next(line for line in text.splitlines() if marker in line)
    assert "| INFO" in line and "churn.test_logging" in line