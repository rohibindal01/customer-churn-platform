"""Download the IBM Telco Customer Churn dataset with the Kaggle CLI."""
from __future__ import annotations

import argparse
import shutil
import subprocess

from churn.config.settings import get_settings
from churn.data.fingerprint import file_sha256
from churn.utils.logger import get_logger

DATASET = "blastchar/telco-customer-churn"
logger = get_logger("download_data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Download again even if the file already exists.",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_directories()
    target = settings.raw_data_path

    if target.exists() and not args.force:
        logger.info("Dataset already present: %s (use --force to re-download)", target)
    else:
        kaggle_exe = shutil.which("kaggle")
        if kaggle_exe is None:
            logger.error("Kaggle CLI not found on PATH. Activate the virtual environment first.")
            return 1

        command = [
            kaggle_exe, "datasets", "download",
            "-d", DATASET, "-p", str(settings.raw_dir), "--unzip",
        ]
        if args.force:
            command.append("--force")

        logger.info("Running: %s", " ".join(command))
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            logger.error("Kaggle download failed with exit code %s", exc.returncode)
            return exc.returncode

    if not target.exists():
        logger.error("Expected file not found after download: %s", target)
        return 1

    size_kb = target.stat().st_size / 1024
    logger.info("Dataset ready: %s (%.1f KB)", target.name, size_kb)
    logger.info("SHA-256: %s", file_sha256(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())