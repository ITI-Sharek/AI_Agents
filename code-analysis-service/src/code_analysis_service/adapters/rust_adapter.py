from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from ..models import AnalysisIssue, StaticAnalysisEvidence


def _resolve(repo_path: str, files: list[str]) -> list[str]:
    return [
        str(Path(repo_path) / f) if not Path(f).is_absolute() else f
        for f in files
    ]


def _parse_clippy_output(output: str, file_paths: list[str]) -> list[AnalysisIssue]:
    issues = []
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("reason") != "compiler-message" and data.get("$message_type") != "diagnostic":
            continue
        msg = data.get("message", {})
        if not isinstance(msg, dict):
            msg = data
        if msg.get("level") not in ("warning", "error"):
            continue
        spans = msg.get("spans", [])
        if not spans:
            continue
        span = spans[0]
        fp = span.get("file_name", "")
        if fp not in file_paths:
            continue
        code = None
        if msg.get("code"):
            code = msg["code"].get("code")
        issues.append(AnalysisIssue(
            line=span.get("line_start", 0),
            column=span.get("column_start", 0),
            severity=msg.get("level", "warning"),
            message=msg.get("message", ""),
            rule_id=code or "clippy::unknown",
            file_path=fp,
            node_type=None,
            heuristic_label=False,
        ))
    return issues


def analyze_rust(
    repo_path: str, file_paths: list[str], timeout: int = 60
) -> StaticAnalysisEvidence:
    if not file_paths:
        return StaticAnalysisEvidence(
            status="no_analyzable_content", language="rust"
        )

    abs_files = _resolve(repo_path, file_paths)

    try:
        result = subprocess.run(
            ["clippy-driver", "--edition", "2021", "--error-format=json"] + abs_files,
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return StaticAnalysisEvidence(status="tool_unavailable", language="rust")
    except subprocess.TimeoutExpired:
        return StaticAnalysisEvidence(status="timeout", language="rust")
    except Exception as exc:
        return StaticAnalysisEvidence(
            status="error", language="rust", error_message=str(exc)
        )

    issues = _parse_clippy_output(result.stderr + "\n" + result.stdout, abs_files)

    if result.returncode != 0 and not issues:
        detail = (result.stderr or "").strip() or (result.stdout or "").strip()
        return StaticAnalysisEvidence(
            status="error",
            language="rust",
            error_message=(
                f"clippy-driver exited with {result.returncode} and produced no "
                f"diagnostics: {detail[:500] or 'no output'}"
            ),
        )

    return StaticAnalysisEvidence(
        status="success",
        language="rust",
        files_analyzed=len(file_paths),
        issues=issues,
    )
