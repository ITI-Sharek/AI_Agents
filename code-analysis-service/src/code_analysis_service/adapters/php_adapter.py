from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..models import AnalysisIssue, StaticAnalysisEvidence


_PHPCS_STANDARD = "Generic"


def _resolve(repo_path: str, files: list[str]) -> list[str]:
    return [
        str(Path(repo_path) / f) if not Path(f).is_absolute() else f
        for f in files
    ]


def analyze_php(
    repo_path: str, file_paths: list[str], timeout: int = 60
) -> StaticAnalysisEvidence:
    if not file_paths:
        return StaticAnalysisEvidence(
            status="no_analyzable_content", language="php"
        )

    abs_files = _resolve(repo_path, file_paths)

    try:
        result = subprocess.run(
            ["phpcs", f"--standard={_PHPCS_STANDARD}", "--report=json"] + abs_files,
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return StaticAnalysisEvidence(status="tool_unavailable", language="php")
    except subprocess.TimeoutExpired:
        return StaticAnalysisEvidence(status="timeout", language="php")
    except Exception as exc:
        return StaticAnalysisEvidence(
            status="error", language="php", error_message=str(exc)
        )

    issues = []

    # PHP_CodeSniffer 3.x exit codes: 0 = clean, 1 = errors found,
    # 2 = fixable issues found (with no errors), 3 = internal error during
    # processing or report generation. Exit 3 must never be interpreted as
    # a successful analysis, even when a partial JSON report was printed.
    if result.returncode == 3:
        detail = (result.stderr or "").strip() or (result.stdout or "").strip()
        return StaticAnalysisEvidence(
            status="error",
            language="php",
            error_message=(
                f"phpcs internal error (exit 3): {detail[:500] or 'no output'}"
            ),
        )

    report = None
    if result.stdout.strip():
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError:
            report = None
        if not isinstance(report, dict):
            report = None
    if report is not None:
        for fp, file_data in report.get("files", {}).items():
            for msg in file_data.get("messages", []):
                issues.append(AnalysisIssue(
                    line=msg.get("line", 0),
                    column=msg.get("column", 0),
                    severity=msg.get("type", "error").lower(),
                    message=msg.get("message", ""),
                    rule_id=msg.get("source", ""),
                    file_path=fp,
                    node_type=None,
                    heuristic_label=False,
                ))
        return StaticAnalysisEvidence(
            status="success",
            language="php",
            files_analyzed=len(file_paths),
            issues=issues,
        )

    if result.returncode != 0:
        detail = (result.stderr or "").strip() or (result.stdout or "").strip()
        return StaticAnalysisEvidence(
            status="error",
            language="php",
            error_message=(
                f"phpcs exited with {result.returncode} and produced no JSON "
                f"report: {detail[:500] or 'no output'}"
            ),
        )

    return StaticAnalysisEvidence(
        status="success",
        language="php",
        files_analyzed=len(file_paths),
        issues=issues,
    )
