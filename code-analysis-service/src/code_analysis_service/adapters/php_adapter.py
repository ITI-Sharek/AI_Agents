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
    try:
        if result.stdout.strip():
            data = json.loads(result.stdout)
            for fp, file_data in data.get("files", {}).items():
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
    except (json.JSONDecodeError, KeyError):
        pass

    return StaticAnalysisEvidence(
        status="success",
        language="php",
        files_analyzed=len(file_paths),
        issues=issues,
    )
