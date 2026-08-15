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


_PARSER_OPTIONS = "ecmaVersion:latest,sourceType:module"


def _parser_args(abs_files: list[str]) -> list[str]:
    args = ["--parser-options", _PARSER_OPTIONS]
    if any(f.endswith((".ts", ".tsx")) for f in abs_files):
        args = ["--parser", "@typescript-eslint/parser"] + args
    return args


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
            ["eslint", "--no-eslintrc", "--format", "json"]
            + _parser_args(abs_files)
            + abs_files,
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
    report_parsed = False
    try:
        if result.stdout.strip():
            data = json.loads(result.stdout)
            report_parsed = True
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

    # ESLint exit codes: 0 = clean, 1 = lint findings, 2 = fatal
    # (parse/config) errors. A parseable JSON report is always treated as
    # the truth — its messages (including fatal parse errors) are preserved
    # as findings, never a false-clean success. Only when ESLint exits
    # non-zero AND produces no parseable JSON report (empty or unparseable
    # stdout) is this an error rather than a clean success.
    if result.returncode != 0 and not report_parsed:
        detail = (result.stderr or "").strip() or (result.stdout or "").strip()
        return StaticAnalysisEvidence(
            status="error",
            language="javascript",
            error_message=(
                f"eslint exited with {result.returncode} and produced no JSON "
                f"report: {detail[:500] or 'no output'}"
            ),
        )

    return StaticAnalysisEvidence(
        status="success",
        language="javascript",
        files_analyzed=len(file_paths),
        issues=issues,
    )
