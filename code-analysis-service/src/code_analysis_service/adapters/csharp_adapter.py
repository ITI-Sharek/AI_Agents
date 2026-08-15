from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ..models import AnalysisIssue, StaticAnalysisEvidence


_DIAGNOSTIC_RE = re.compile(
    r"^\s*(.+?)\((\d+),(\d+)\)\s*:\s*(warning|error)\s+(\w+):\s+(.+)$"
)


def _resolve(repo_path: str, files: list[str]) -> list[str]:
    return [
        str(Path(repo_path) / f) if not Path(f).is_absolute() else f
        for f in files
    ]


def _find_project(repo_path: str) -> str | None:
    path = Path(repo_path)
    csproj_files = list(path.rglob("*.csproj"))
    if csproj_files:
        return str(csproj_files[0])
    sln_files = list(path.rglob("*.sln"))
    if sln_files:
        return str(sln_files[0])
    return None


def _parse_diagnostics(
    output: str, abs_files: list[str]
) -> list[AnalysisIssue]:
    issues = []
    abs_set = set(abs_files)
    for line in output.splitlines():
        match = _DIAGNOSTIC_RE.match(line)
        if not match:
            continue
        fp = match.group(1)
        resolved = str(Path(fp).resolve())
        if resolved not in abs_set:
            continue
        issues.append(AnalysisIssue(
            line=int(match.group(2)),
            column=int(match.group(3)),
            severity=match.group(4),
            message=match.group(6),
            rule_id=match.group(5),
            file_path=fp,
            node_type=None,
            heuristic_label=False,
        ))
    return issues


def analyze_csharp(
    repo_path: str, file_paths: list[str], timeout: int = 60
) -> StaticAnalysisEvidence:
    if not file_paths:
        return StaticAnalysisEvidence(
            status="no_analyzable_content", language="csharp"
        )

    abs_files = _resolve(repo_path, file_paths)

    project_file = _find_project(repo_path)
    if not project_file:
        return StaticAnalysisEvidence(
            status="error",
            language="csharp",
            error_message="no .csproj or .sln file found in repo_path",
        )

    try:
        result = subprocess.run(
            [
                "dotnet", "build", project_file,
                "--no-restore",
                "-p:TreatWarningsAsErrors=false",
            ],
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        return StaticAnalysisEvidence(status="tool_unavailable", language="csharp")
    except subprocess.TimeoutExpired:
        return StaticAnalysisEvidence(status="timeout", language="csharp")
    except Exception as exc:
        return StaticAnalysisEvidence(
            status="error", language="csharp", error_message=str(exc)
        )

    issues = _parse_diagnostics(result.stderr + "\n" + result.stdout, abs_files)

    if result.returncode != 0 and not issues:
        detail = (result.stderr or "").strip() or (result.stdout or "").strip()
        return StaticAnalysisEvidence(
            status="error",
            language="csharp",
            error_message=(
                f"dotnet build exited with {result.returncode} and produced no "
                f"diagnostics: {detail[:500] or 'no output'}"
            ),
        )

    return StaticAnalysisEvidence(
        status="success",
        language="csharp",
        files_analyzed=len(file_paths),
        issues=issues,
    )
