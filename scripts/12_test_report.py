"""Milestone 10: turn the pytest JUnit XML and the coverage XML into docs\\test_report.md.

Run the test suite first:
  python -m pytest tests --cov --cov-report=term-missing
      --cov-report=xml:artifacts\\tests\\coverage.xml --junitxml=artifacts\\tests\\junit.xml
Then:
  python scripts\\12_test_report.py
"""
from __future__ import annotations

import platform
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata

from churn.config.settings import get_settings
from churn.utils.logger import get_logger

logger = get_logger("test_report")
PACKAGES = ["pytest", "pytest-cov", "scikit-learn", "imbalanced-learn", "xgboost", "lightgbm",
            "catboost", "shap", "pandas", "numpy", "streamlit", "mlflow"]
COMMAND = ("python -m pytest tests --cov --cov-report=term-missing "
           "--cov-report=xml:artifacts\\tests\\coverage.xml --junitxml=artifacts\\tests\\junit.xml")


def parse_junit(path) -> dict:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    cases = []
    for suite in suites:
        for case in suite.iter("testcase"):
            status, message = "passed", ""
            for tag, label in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
                node = case.find(tag)
                if node is not None:
                    status, message = label, (node.get("message") or "").strip()
                    break
            classname = case.get("classname", "")
            cases.append({"file": classname.replace(".", "/") + ".py",
                          "name": case.get("name", ""),
                          "seconds": float(case.get("time") or 0.0),
                          "status": status, "message": message})
    timestamp = suites[0].get("timestamp", "") if suites else ""
    return {"cases": cases, "timestamp": timestamp}


def parse_coverage(path) -> dict:
    root = ET.parse(path).getroot()
    files = []
    for cls in root.iter("class"):
        lines = cls.find("lines")
        hits = [int(line.get("hits", 0)) for line in lines.iter("line")] if lines is not None else []
        files.append({"file": cls.get("filename", ""), "lines": len(hits),
                      "covered": sum(1 for h in hits if h > 0),
                      "line_rate": float(cls.get("line-rate", 0.0)),
                      "branch_rate": float(cls.get("branch-rate", 0.0))})
    return {"line_rate": float(root.get("line-rate", 0.0)),
            "branch_rate": float(root.get("branch-rate", 0.0)), "files": files}


def _versions() -> list[str]:
    found = []
    for name in PACKAGES:
        try:
            found.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            found.append(f"{name} (not installed)")
    return found


def build_report(junit: dict, coverage: dict | None) -> tuple[str, dict]:
    cases = junit["cases"]
    totals = defaultdict(int)
    for case in cases:
        totals[case["status"]] += 1
    duration = sum(c["seconds"] for c in cases)
    summary = {"tests": len(cases), "passed": totals["passed"], "failed": totals["failed"],
               "errors": totals["error"], "skipped": totals["skipped"], "seconds": duration}

    lines = [
        "# Test report",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from an actual "
        f"test run started {junit['timestamp'] or 'unknown'}.",
        "",
        f"Command: `{COMMAND}`",
        "",
        "## Summary",
        "",
        "| Tests | Passed | Failed | Errors | Skipped | Duration |",
        "|---|---|---|---|---|---|",
        f"| {summary['tests']} | {summary['passed']} | {summary['failed']} | {summary['errors']} "
        f"| {summary['skipped']} | {duration:.0f} s |",
        "",
        "Skipped tests are not passed tests: they did not run (see the reasons below).",
        "",
        "## Results by test file",
        "",
        "| File | Tests | Passed | Failed | Skipped | Seconds |",
        "|---|---|---|---|---|---|",
    ]
    by_file = defaultdict(list)
    for case in cases:
        by_file[case["file"]].append(case)
    for file in sorted(by_file):
        group = by_file[file]
        count = lambda status: sum(c["status"] == status for c in group)  # noqa: E731
        lines.append(f"| `{file}` | {len(group)} | {count('passed')} | "
                     f"{count('failed') + count('error')} | {count('skipped')} | "
                     f"{sum(c['seconds'] for c in group):.1f} |")

    problems = [c for c in cases if c["status"] in ("failed", "error")]
    if problems:
        lines += ["", "## Failures", ""]
        lines += [f"- `{c['file']}::{c['name']}`: {c['message'][:300]}" for c in problems]
    skipped = [c for c in cases if c["status"] == "skipped"]
    if skipped:
        lines += ["", "## Skipped tests and why", ""]
        lines += [f"- `{c['file']}::{c['name']}`: {c['message'][:200]}" for c in skipped]

    lines += ["", "## Coverage", ""]
    if coverage is None:
        lines.append("No coverage file found.")
    else:
        lines += [
            f"Overall line coverage **{coverage['line_rate']:.1%}**, branch coverage "
            f"**{coverage['branch_rate']:.1%}** (source: `src/churn` and `app`).",
            "",
            "Least covered first. Coverage shows what did not run; it does not measure test quality.",
            "",
            "| File | Lines | Covered | Line coverage | Branch coverage |",
            "|---|---|---|---|---|",
        ]
        for f in sorted(coverage["files"], key=lambda f: (f["line_rate"], f["file"])):
            lines.append(f"| `{f['file']}` | {f['lines']} | {f['covered']} | "
                         f"{f['line_rate']:.1%} | {f['branch_rate']:.1%} |")

    lines += ["", "## Environment", "",
              f"- Python {sys.version.split()[0]} on {platform.platform()}",
              *[f"- {v}" for v in _versions()], ""]
    return "\n".join(lines), summary


def main() -> int:
    settings = get_settings()
    junit_path = settings.artifacts_dir / "tests" / "junit.xml"
    coverage_path = settings.artifacts_dir / "tests" / "coverage.xml"
    if not junit_path.exists():
        logger.error("No %s. Run the test command first:\n  %s", junit_path, COMMAND)
        return 1
    junit = parse_junit(junit_path)
    coverage = parse_coverage(coverage_path) if coverage_path.exists() else None
    report, summary = build_report(junit, coverage)

    docs_dir = settings.root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / "test_report.md"
    out.write_text(report, encoding="utf-8")
    logger.info("Tests: %d | passed %d | failed %d | errors %d | skipped %d | %.0f s",
                summary["tests"], summary["passed"], summary["failed"], summary["errors"],
                summary["skipped"], summary["seconds"])
    if coverage is not None:
        logger.info("Coverage: lines %.1f%%, branches %.1f%%",
                    coverage["line_rate"] * 100, coverage["branch_rate"] * 100)
    logger.info("Wrote %s", out)
    return 0 if summary["failed"] == 0 and summary["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())