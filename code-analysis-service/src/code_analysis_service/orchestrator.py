from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import time
from typing import Any, Literal

from .container_sandbox import (
    check_docker,
    clone_in_container,
    run_analysis_in_container,
)
from .models import (
    AnalysisIssue,
    AnalysisResult,
    CircularImport,
    CloneResult,
    GraphEdge,
    GraphNode,
    GraphRelationsEvidence,
    InheritanceRelation,
    StaticAnalysisEvidence,
    StructuralGraph,
)

logger = logging.getLogger(__name__)

_DETERMINISTIC_STATUSES = frozenset({
    "language_not_supported",
    "no_analyzable_content",
    "repo_too_large",
    "tool_unavailable",
    "authentication_failed",
})

_SANDBOX_UNAVAILABLE_MSG = (
    "sandbox unavailable: docker is not available on this host — "
    "refusing host-side execution"
)


def _classify_failure(status: str, error_message: str | None = None) -> str:
    if status == "success":
        return "success"
    if status in _DETERMINISTIC_STATUSES:
        return "deterministic"
    if status == "timeout":
        return "transient"
    if status == "clone_failed":
        if error_message and "timed out" in error_message.lower():
            return "transient"
        return "deterministic"
    if status == "error":
        return "deterministic"
    return "deterministic"


def _static_evidence_from_dict(data: dict[str, Any]) -> StaticAnalysisEvidence:
    structure_data = data.get("structure") or {}
    structure = StructuralGraph(
        inheritance_relationships=[
            InheritanceRelation(**r)
            for r in structure_data.get("inheritance_relationships", [])
        ],
        coupling=structure_data.get("coupling"),
        circular_imports=[
            CircularImport(**c)
            for c in structure_data.get("circular_imports", [])
        ],
    )
    return StaticAnalysisEvidence(
        status=data.get("status", "success"),
        language=data.get("language", ""),
        files_analyzed=data.get("files_analyzed", 0),
        complexity=data.get("complexity"),
        maintainability_index=data.get("maintainability_index"),
        issues=[AnalysisIssue(**i) for i in data.get("issues", [])],
        structure=structure,
        error_message=data.get("error_message"),
    )


def _graph_evidence_from_dict(data: dict[str, Any]) -> GraphRelationsEvidence:
    return GraphRelationsEvidence(
        status=data.get("status", "success"),
        nodes=[GraphNode(**n) for n in data.get("nodes", [])],
        edges=[GraphEdge(**e) for e in data.get("edges", [])],
        inheritance_depth=data.get("inheritance_depth"),
        coupling=data.get("coupling"),
        coupling_summary=data.get("coupling_summary", ""),
        circular_imports=[
            CircularImport(**c) for c in data.get("circular_imports", [])
        ],
        error_message=data.get("error_message"),
    )


async def _clone_repo(
    repo_url: str,
    pat: str | None,
    dest_dir: str,
    timeout_seconds: int,
) -> CloneResult:
    """Clone inside the runner container.

    The sandbox is MANDATORY: this is the only clone path used by
    ``/analyze/repo``. The host-side ``clone_repo()`` in ``clone.py`` is
    never called by the orchestrator.
    """
    return await clone_in_container(
        repo_url=repo_url, pat=pat, dest_dir=dest_dir,
        timeout_seconds=timeout_seconds,
    )


async def _run_static_analysis(
    language: str,
    repo_path: str,
    budget: int,
) -> StaticAnalysisEvidence:
    try:
        returncode, stdout, stderr = await run_analysis_in_container(
            tool="static_analysis",
            language=language,
            work_dir=repo_path,
            timeout=budget,
        )
    except asyncio.TimeoutError:
        return StaticAnalysisEvidence(status="timeout", language=language)
    except Exception as exc:
        logger.warning("static analysis sandbox run failed: %s", exc)
        return StaticAnalysisEvidence(
            status="tool_unavailable",
            language=language,
            error_message="sandbox execution failed",
        )

    if returncode != 0:
        logger.warning(
            "static analysis sandbox run exited %s: %s", returncode, stderr[:200]
        )
        return StaticAnalysisEvidence(
            status="tool_unavailable",
            language=language,
            error_message="sandbox execution failed",
        )
    try:
        return _static_evidence_from_dict(json.loads(stdout or "{}"))
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("cannot parse sandbox static-analysis output: %s", exc)
        return StaticAnalysisEvidence(
            status="tool_unavailable",
            language=language,
            error_message="sandbox execution failed",
        )


async def _run_graph_relations(
    repo_path: str,
    budget: int,
) -> GraphRelationsEvidence:
    try:
        returncode, stdout, stderr = await run_analysis_in_container(
            tool="graph_relations",
            language="",
            work_dir=repo_path,
            timeout=budget,
        )
    except asyncio.TimeoutError:
        return GraphRelationsEvidence(status="timeout")
    except Exception as exc:
        logger.warning("graph sandbox run failed: %s", exc)
        return GraphRelationsEvidence(
            status="tool_unavailable",
            error_message="sandbox execution failed",
        )

    if returncode != 0:
        logger.warning(
            "graph sandbox run exited %s: %s", returncode, stderr[:200]
        )
        return GraphRelationsEvidence(
            status="tool_unavailable",
            error_message="sandbox execution failed",
        )
    try:
        return _graph_evidence_from_dict(json.loads(stdout or "{}"))
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("cannot parse sandbox graph output: %s", exc)
        return GraphRelationsEvidence(
            status="tool_unavailable",
            error_message="sandbox execution failed",
        )


async def _run_tool(
    tool: str,
    language: str,
    repo_path: str,
    budget: int,
) -> StaticAnalysisEvidence | GraphRelationsEvidence:
    if tool == "static_analysis":
        return await _run_static_analysis(language, repo_path, budget)
    if tool == "graph_relations":
        return await _run_graph_relations(repo_path, budget)
    return StaticAnalysisEvidence(
        status="language_not_supported", language=language
    )


def _set_result(
    result: AnalysisResult,
    tool: str,
    evidence: StaticAnalysisEvidence | GraphRelationsEvidence,
) -> None:
    if tool == "static_analysis":
        result.static_analysis = evidence
    elif tool == "graph_relations":
        result.graph_relations = evidence


def _exhausted_evidence(
    tool: str, language: str,
) -> StaticAnalysisEvidence | GraphRelationsEvidence:
    if tool == "static_analysis":
        return StaticAnalysisEvidence(
            status="transient_failure_exhausted", language=language
        )
    return GraphRelationsEvidence(status="transient_failure_exhausted")


def _clone_to_evidence(
    tool: str, language: str, clone_status: str,
) -> StaticAnalysisEvidence | GraphRelationsEvidence:
    if tool == "static_analysis":
        return StaticAnalysisEvidence(status=clone_status, language=language)
    return GraphRelationsEvidence(status=clone_status)


def _sandbox_unavailable_evidence(
    tool: str, language: str,
) -> StaticAnalysisEvidence | GraphRelationsEvidence:
    if tool == "static_analysis":
        return StaticAnalysisEvidence(
            status="tool_unavailable",
            language=language,
            error_message=_SANDBOX_UNAVAILABLE_MSG,
        )
    return GraphRelationsEvidence(
        status="tool_unavailable",
        error_message=_SANDBOX_UNAVAILABLE_MSG,
    )


async def analyze_repo(
    repo_url: str,
    pat: str | None,
    language: str,
    requested_tools: list[Literal["static_analysis", "graph_relations"]],
) -> AnalysisResult:
    """Analyze a repository — sandbox execution is MANDATORY.

    Every clone and every static-analysis/Graphify run executes inside
    the runner container. If the sandbox (Docker) is unavailable, the
    request fails closed with ``tool_unavailable`` — the service NEVER
    falls back to host-side ``clone_repo()`` or host-side tool execution.
    """
    result = AnalysisResult()
    remaining = set(requested_tools)

    if not check_docker():
        logger.warning(
            "Docker is unavailable — failing closed with tool_unavailable; "
            "no host-side execution will be attempted"
        )
        for tool in requested_tools:
            _set_result(result, tool, _sandbox_unavailable_evidence(tool, language))
        return result

    logger.info("Docker available — running clone and analysis inside the container sandbox")

    first_dir = tempfile.mkdtemp(prefix="code-analysis-attempt1-")
    try:
        start = time.monotonic()
        clone_result = await _clone_repo(
            repo_url, pat, first_dir, 90,
        )
        elapsed = time.monotonic() - start

        if clone_result.status == "success":
            logger.info("clone succeeded in %.1fs", elapsed)
            for tool in list(remaining):
                tool_budget = max(10, 90 - int(elapsed))
                evidence = await _run_tool(tool, language, first_dir, tool_budget)
                cat = _classify_failure(evidence.status, getattr(evidence, "error_message", None))
                if cat == "success":
                    logger.info("first attempt: %s succeeded", tool)
                    _set_result(result, tool, evidence)
                    remaining.discard(tool)
                elif cat == "deterministic":
                    logger.info(
                        "first attempt: %s deterministic status=%s — no retry",
                        tool, evidence.status,
                    )
                    _set_result(result, tool, evidence)
                    remaining.discard(tool)
                else:
                    logger.info(
                        "first attempt: %s transient status=%s — will retry",
                        tool, evidence.status,
                    )
        else:
            cat = _classify_failure(clone_result.status, clone_result.error_message)
            logger.info(
                "clone status=%s category=%s elapsed=%.1fs",
                clone_result.status, cat, elapsed,
            )
            if cat == "deterministic":
                logger.info("deterministic clone failure — no retry for any tool")
                for tool in remaining:
                    _set_result(
                        result, tool,
                        _clone_to_evidence(tool, language, clone_result.status),
                    )
                return result
            logger.info("transient clone failure — will retry all tools")
    finally:
        shutil.rmtree(first_dir, ignore_errors=True)

    if not remaining:
        return result

    logger.info("retrying %d tool(s) with 180s budget", len(remaining))
    second_dir = tempfile.mkdtemp(prefix="code-analysis-attempt2-")
    try:
        start = time.monotonic()
        clone_result = await _clone_repo(
            repo_url, pat, second_dir, 180,
        )
        elapsed = time.monotonic() - start

        if clone_result.status != "success":
            logger.warning("retry: clone failed — exhausted for all tools")
            for tool in remaining:
                _set_result(result, tool, _exhausted_evidence(tool, language))
            return result

        for tool in list(remaining):
            tool_budget = max(10, 180 - int(elapsed))
            evidence = await _run_tool(tool, language, second_dir, tool_budget)

            if evidence.status == "success":
                logger.info("retry: %s succeeded", tool)
                _set_result(result, tool, evidence)
            else:
                logger.warning(
                    "retry: %s status=%s — exhausted",
                    tool, evidence.status,
                )
                _set_result(result, tool, _exhausted_evidence(tool, language))
    finally:
        shutil.rmtree(second_dir, ignore_errors=True)

    return result
