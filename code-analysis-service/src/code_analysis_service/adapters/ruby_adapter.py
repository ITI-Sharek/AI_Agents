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


def analyze_ruby(
    repo_path: str, file_paths: list[str], timeout: int = 60
) -> StaticAnalysisEvidence:
    if not file_paths:
        return StaticAnalysisEvidence(
            status="no_analyzable_content", language="ruby"
        )

    abs_files = _resolve(repo_path, file_paths)

    try:
        result = subprocess.run(
            ["rubocop", "--force-default-config", "--format", "json"] + abs_files,
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return StaticAnalysisEvidence(status="tool_unavailable", language="ruby")
    except subprocess.TimeoutExpired:
        return StaticAnalysisEvidence(status="timeout", language="ruby")
    except Exception as exc:
        return StaticAnalysisEvidence(
            status="error", language="ruby", error_message=str(exc)
        )

    issues = []
    try:
        if result.stdout.strip():
            data = json.loads(result.stdout)
            for entry in data.get("files", []):
                fp = entry.get("path", "")
                for off in entry.get("offenses", []):
                    loc = off.get("location", {})
                    issues.append(AnalysisIssue(
                        line=loc.get("line", 0),
                        column=loc.get("column", 0),
                        severity=off.get("severity", "convention"),
                        message=off.get("message", ""),
                        rule_id=off.get("cop_name", ""),
                        file_path=fp,
                        node_type=None,
                        heuristic_label=False,
                    ))
    except (json.JSONDecodeError, KeyError):
        pass

    return StaticAnalysisEvidence(
        status="success",
        language="ruby",
        files_analyzed=len(file_paths),
        issues=issues,
    )
