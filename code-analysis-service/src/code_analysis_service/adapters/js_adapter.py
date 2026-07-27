from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..models import AnalysisIssue, StaticAnalysisEvidence


def _resolve(repo_path: str, files: list[str]) -> list[str]:
    return [
        str(Path(repo_path) / f) if not Path(f).is_absolute() else f
        for f in files
    ]


def analyze_js(
    repo_path: str, file_paths: list[str], timeout: int = 60
) -> StaticAnalysisEvidence:
    if not file_paths:
        return StaticAnalysisEvidence(
            status="no_analyzable_content", language="javascript"
        )

    abs_files = _resolve(repo_path, file_paths)

    try:
        result = subprocess.run(
            ["eslint", "--no-eslintrc", "--format", "json"] + abs_files,
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return StaticAnalysisEvidence(status="tool_unavailable", language="javascript")
    except subprocess.TimeoutExpired:
        return StaticAnalysisEvidence(status="timeout", language="javascript")
    except Exception as exc:
        return StaticAnalysisEvidence(
            status="error", language="javascript", error_message=str(exc)
        )

    issues = []
    try:
        if result.stdout.strip():
            data = json.loads(result.stdout)
            for entry in data:
                fp = entry.get("filePath", "")
                for msg in entry.get("messages", []):
                    severity_map = {1: "warning", 2: "error"}
                    sev = severity_map.get(msg.get("severity", 0), "info")
                    node_type = msg.get("nodeType")
                    issues.append(AnalysisIssue(
                        line=msg.get("line", 0),
                        column=msg.get("column", 0),
                        severity=sev,
                        message=msg.get("message", ""),
                        rule_id=msg.get("ruleId", ""),
                        file_path=fp,
                        node_type=node_type,
                        heuristic_label=False,
                    ))
    except (json.JSONDecodeError, KeyError):
        pass

    return StaticAnalysisEvidence(
        status="success",
        language="javascript",
        files_analyzed=len(file_paths),
        issues=issues,
    )
