from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path
from typing import Optional

from ..models import (
    AnalysisIssue,
    InheritanceRelation,
    StaticAnalysisEvidence,
    StructuralGraph,
)


def _resolve(repo_path: str, files: list[str]) -> list[str]:
    return [
        str(Path(repo_path) / f) if not Path(f).is_absolute() else f
        for f in files
    ]


def _run_radon_cc(
    files: list[str], timeout: int
) -> tuple[Optional[float], list[dict]]:
    result = subprocess.run(
        ["radon", "cc", "-j"] + files,
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None, []
    data = json.loads(result.stdout)
    values = []
    for funcs in data.values():
        for f in funcs:
            values.append(f["complexity"])
    avg = sum(values) / len(values) if values else None
    return avg, []


def _run_radon_mi(files: list[str], timeout: int) -> Optional[float]:
    result = subprocess.run(
        ["radon", "mi", "-j"] + files,
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    data = json.loads(result.stdout)
    values = []
    for file_data in data.values():
        mi = file_data.get("mi") if isinstance(file_data, dict) else None
        if mi is not None:
            values.append(mi)
    return sum(values) / len(values) if values else None


def _run_pylint(files: list[str], timeout: int) -> list[AnalysisIssue]:
    result = subprocess.run(
        ["pylint", "--output-format=json", "--rcfile=/dev/null"] + files,
        capture_output=True, text=True, timeout=timeout,
    )
    if not result.stdout.strip():
        return []
    data = json.loads(result.stdout)
    issues = []
    for item in data:
        issues.append(AnalysisIssue(
            line=item["line"],
            column=item.get("column", 0),
            severity=item["type"],
            message=item["message"],
            rule_id=item["symbol"],
            file_path=item["path"],
            node_type=item.get("obj") or None,
            heuristic_label=False,
        ))
    return issues


def _detect_inheritance(files: list[str]) -> list[InheritanceRelation]:
    relations = []
    for fp in files:
        try:
            with open(fp) as f:
                source = f.read()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        name = None
                        if isinstance(base, ast.Name):
                            name = base.id
                        elif isinstance(base, ast.Attribute):
                            name = base.attr
                        if name:
                            relations.append(InheritanceRelation(
                                child_class=node.name,
                                parent_class=name,
                                file_path=fp,
                                line=node.lineno,
                            ))
        except (IOError, SyntaxError):
            pass
    return relations


def analyze_python(
    repo_path: str, file_paths: list[str], timeout: int = 60
) -> StaticAnalysisEvidence:
    if not file_paths:
        return StaticAnalysisEvidence(
            status="no_analyzable_content", language="python"
        )

    abs_files = _resolve(repo_path, file_paths)

    try:
        avg_complexity, _ = _run_radon_cc(abs_files, timeout)
    except FileNotFoundError:
        return StaticAnalysisEvidence(status="tool_unavailable", language="python")
    except subprocess.TimeoutExpired:
        return StaticAnalysisEvidence(status="timeout", language="python")
    except Exception as exc:
        return StaticAnalysisEvidence(
            status="error", language="python", error_message=str(exc)
        )

    try:
        maintainability = _run_radon_mi(abs_files, timeout)
    except FileNotFoundError:
        return StaticAnalysisEvidence(status="tool_unavailable", language="python")
    except subprocess.TimeoutExpired:
        return StaticAnalysisEvidence(status="timeout", language="python")
    except Exception as exc:
        return StaticAnalysisEvidence(
            status="error", language="python", error_message=str(exc)
        )

    try:
        pylint_issues = _run_pylint(abs_files, timeout)
    except FileNotFoundError:
        return StaticAnalysisEvidence(status="tool_unavailable", language="python")
    except subprocess.TimeoutExpired:
        return StaticAnalysisEvidence(status="timeout", language="python")
    except Exception as exc:
        return StaticAnalysisEvidence(
            status="error", language="python", error_message=str(exc)
        )

    inheritance = _detect_inheritance(abs_files)

    return StaticAnalysisEvidence(
        status="success",
        language="python",
        files_analyzed=len(file_paths),
        complexity=avg_complexity,
        maintainability_index=maintainability,
        issues=pylint_issues,
        structure=StructuralGraph(inheritance_relationships=inheritance),
    )
