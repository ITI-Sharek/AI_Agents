from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Optional

from ..models import AnalysisIssue, StaticAnalysisEvidence


def _resolve(repo_path: str, files: list[str]) -> list[str]:
    return [
        str(Path(repo_path) / f) if not Path(f).is_absolute() else f
        for f in files
    ]


def _run_gocyclo(files: list[str], timeout: int) -> Optional[float]:
    result = subprocess.run(
        ["gocyclo", "-avg"] + files,
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        return None
    match = re.search(r"Avg:\s+([\d.]+)", result.stdout)
    if match:
        return float(match.group(1))
    return None


def _run_golangci_lint(files: list[str], timeout: int) -> list[AnalysisIssue]:
    result = subprocess.run(
        ["golangci-lint", "run", "--no-config", "--out-format=json"] + files,
        capture_output=True, text=True, timeout=timeout,
    )
    issues = []
    if result.stdout.strip():
        try:
            data = json.loads(result.stdout)
            for issue in data.get("Issues", []):
                pos = issue.get("Pos", {})
                issues.append(AnalysisIssue(
                    line=pos.get("Line", 0),
                    column=pos.get("Column", 0),
                    severity=issue.get("Severity", "warning"),
                    message=issue.get("Text", ""),
                    rule_id=issue.get("FromLinter", ""),
                    file_path=pos.get("Filename", ""),
                    node_type=None,
                    heuristic_label=False,
                ))
        except (json.JSONDecodeError, KeyError):
            pass
    return issues


def analyze_go(
    repo_path: str, file_paths: list[str], timeout: int = 60
) -> StaticAnalysisEvidence:
    if not file_paths:
        return StaticAnalysisEvidence(status="no_analyzable_content", language="go")

    abs_files = _resolve(repo_path, file_paths)

    try:
        avg_complexity = _run_gocyclo(abs_files, timeout)
    except FileNotFoundError:
        return StaticAnalysisEvidence(status="tool_unavailable", language="go")
    except subprocess.TimeoutExpired:
        return StaticAnalysisEvidence(status="timeout", language="go")
    except Exception as exc:
        return StaticAnalysisEvidence(
            status="error", language="go", error_message=str(exc)
        )

    try:
        lint_issues = _run_golangci_lint(abs_files, timeout)
    except FileNotFoundError:
        return StaticAnalysisEvidence(status="tool_unavailable", language="go")
    except subprocess.TimeoutExpired:
        return StaticAnalysisEvidence(status="timeout", language="go")
    except Exception as exc:
        return StaticAnalysisEvidence(
            status="error", language="go", error_message=str(exc)
        )

    return StaticAnalysisEvidence(
        status="success",
        language="go",
        files_analyzed=len(file_paths),
        complexity=avg_complexity,
        issues=lint_issues,
    )
