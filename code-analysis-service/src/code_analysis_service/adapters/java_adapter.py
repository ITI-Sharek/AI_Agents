from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from ..models import AnalysisIssue, StaticAnalysisEvidence


_CHECKSTYLE_CONFIG = "/sun_checks.xml"


def _resolve(repo_path: str, files: list[str]) -> list[str]:
    return [
        str(Path(repo_path) / f) if not Path(f).is_absolute() else f
        for f in files
    ]


def _parse_checkstyle_xml(xml_content: str) -> list[AnalysisIssue]:
    issues = []
    try:
        root = ET.fromstring(xml_content)
        for file_elem in root.findall("file"):
            fp = file_elem.get("name", "")
            for err in file_elem.findall("error"):
                severity = err.get("severity", "error")
                source = err.get("source", "")
                rule_id = source.split(".")[-1] if source else ""
                issues.append(AnalysisIssue(
                    line=int(err.get("line", 0)),
                    column=int(err.get("column", 0)),
                    severity=severity,
                    message=err.get("message", ""),
                    rule_id=rule_id,
                    file_path=fp,
                    node_type=None,
                    heuristic_label=False,
                ))
    except ET.ParseError:
        pass
    return issues


def analyze_java(
    repo_path: str, file_paths: list[str], timeout: int = 60
) -> StaticAnalysisEvidence:
    if not file_paths:
        return StaticAnalysisEvidence(
            status="no_analyzable_content", language="java"
        )

    abs_files = _resolve(repo_path, file_paths)

    try:
        result = subprocess.run(
            ["checkstyle", "-c", _CHECKSTYLE_CONFIG, "-f", "xml"] + abs_files,
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return StaticAnalysisEvidence(status="tool_unavailable", language="java")
    except subprocess.TimeoutExpired:
        return StaticAnalysisEvidence(status="timeout", language="java")
    except Exception as exc:
        return StaticAnalysisEvidence(
            status="error", language="java", error_message=str(exc)
        )

    issues = _parse_checkstyle_xml(result.stdout)

    return StaticAnalysisEvidence(
        status="success",
        language="java",
        files_analyzed=len(file_paths),
        issues=issues,
    )
