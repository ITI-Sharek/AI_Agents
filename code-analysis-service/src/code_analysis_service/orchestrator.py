from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Literal

from .container_sandbox import (
    _REPO_DIR,
    _TIMED_OUT_EXIT_CODE,
    check_docker,
    clone_in_container,
    disconnect_analysis_container,
    is_workspace_available,
    remove_analysis_container,
    run_analysis_in_container,
    start_analysis_container,
)
from .models import (
    AnalysisIssue,
    AnalysisResult,
    CircularImport,
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


async def _run_static_analysis(
    container: str,
    language: str,
    repo_dir: str,
    budget: int,
) -> StaticAnalysisEvidence:
    try:
        returncode, stdout, stderr = await run_analysis_in_container(
            container=container,
            tool="static_analysis",
            language=language,
            repo_dir=repo_dir,
            timeout=budget,
        )
    except asyncio.TimeoutError:
        # The in-container watchdog should have reported 124/125 within
        # the exec margin; reaching this point means the docker exec client
        # itself hung and the analyzer process-group state is UNKNOWN.
        # Fail closed: never retry against a possibly-live analyzer.
        logger.warning(
            "static analysis exec client timed out — analyzer process state "
            "unverified, failing closed"
        )
        return StaticAnalysisEvidence(
            status="tool_unavailable",
            language=language,
            error_message="sandbox execution failed",
        )
    except Exception as exc:
        logger.warning("static analysis sandbox run failed: %s", exc)
        return StaticAnalysisEvidence(
            status="tool_unavailable",
            language=language,
            error_message="sandbox execution failed",
        )

    if returncode == _TIMED_OUT_EXIT_CODE:
        logger.warning("static analysis sandbox run timed out")
        return StaticAnalysisEvidence(status="timeout", language=language)
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
    container: str,
    repo_dir: str,
    budget: int,
) -> GraphRelationsEvidence:
    try:
        returncode, stdout, stderr = await run_analysis_in_container(
            container=container,
            tool="graph_relations",
            language="",
            repo_dir=repo_dir,
            timeout=budget,
        )
    except asyncio.TimeoutError:
        # Same fail-closed rule as static analysis: the exec client hung,
        # the Graphify process-group state is UNKNOWN, so never retry.
        logger.warning(
            "graph exec client timed out — graphify process state "
            "unverified, failing closed"
        )
        return GraphRelationsEvidence(
            status="tool_unavailable",
            error_message="sandbox execution failed",
        )
    except Exception as exc:
        logger.warning("graph sandbox run failed: %s", exc)
        return GraphRelationsEvidence(
            status="tool_unavailable",
            error_message="sandbox execution failed",
        )

    if returncode == _TIMED_OUT_EXIT_CODE:
        logger.warning("graph sandbox run timed out")
        return GraphRelationsEvidence(status="timeout")
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
    container: str,
    tool: str,
    language: str,
    repo_dir: str,
    budget: int,
) -> StaticAnalysisEvidence | GraphRelationsEvidence:
    if tool == "static_analysis":
        return await _run_static_analysis(container, language, repo_dir, budget)
    if tool == "graph_relations":
        return await _run_graph_relations(container, repo_dir, budget)
    return StaticAnalysisEvidence(
        status="language_not_supported", language=language
    )


async def _run_tools_parallel(
    container: str,
    language: str,
    repo_dir: str,
    tools: list[str],
    budget: int,
) -> dict[str, StaticAnalysisEvidence | GraphRelationsEvidence]:
    """Run every requested tool concurrently inside the SAME container.

    Static Analysis and Graphify are scheduled with ``asyncio.create_task``
    and awaited via ``asyncio.gather``, so both ``docker exec`` runs overlap
    within one per-repository container.
    """
    tasks = [
        (
            tool,
            asyncio.create_task(
                _run_tool(container, tool, language, repo_dir, budget)
            ),
        )
        for tool in tools
    ]
    outcomes: dict[str, StaticAnalysisEvidence | GraphRelationsEvidence] = {}
    results = await asyncio.gather(
        *(task for _, task in tasks), return_exceptions=True
    )
    for (tool, _), outcome in zip(tasks, results):
        if isinstance(outcome, BaseException):
            logger.warning("%s task failed unexpectedly: %s", tool, outcome)
            outcomes[tool] = _sandbox_unavailable_evidence(tool, language)
        else:
            outcomes[tool] = outcome
    return outcomes


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


def _isolation_failed_evidence(
    tool: str, language: str,
) -> StaticAnalysisEvidence | GraphRelationsEvidence:
    if tool == "static_analysis":
        return StaticAnalysisEvidence(
            status="transient_failure_exhausted",
            language=language,
            error_message="network isolation of the analysis sandbox failed",
        )
    return GraphRelationsEvidence(
        status="transient_failure_exhausted",
        error_message="network isolation of the analysis sandbox failed",
    )


async def analyze_repo(
    repo_url: str,
    pat: str | None,
    language: str,
    requested_tools: list[Literal["static_analysis", "graph_relations"]],
) -> AnalysisResult:
    """Analyze a repository — sandbox execution is MANDATORY.

    ONE persistent container is started for the whole request: the repo is
    cloned once into its ``/workspace`` tmpfs while the container is on a
    dedicated network, then the container is DISCONNECTED from that
    network so Static Analysis and Graphify run with NO network access
    (loopback only). Both tool runs execute inside that same container via
    ``docker exec``, concurrently. If the sandbox (Docker) is unavailable,
    the request fails closed with ``tool_unavailable`` — the service NEVER
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

    container: str | None = None
    network: str | None = None
    try:
        container, network = await start_analysis_container()
    except Exception as exc:
        logger.warning("cannot start analysis container: %s", exc)
        for tool in requested_tools:
            _set_result(result, tool, _sandbox_unavailable_evidence(tool, language))
        return result

    try:
        # ---------------- attempt 1 ----------------
        start = time.monotonic()
        clone_result = await clone_in_container(
            container, repo_url, pat, timeout_seconds=90
        )
        elapsed = time.monotonic() - start

        if clone_result.status == "success":
            logger.info("clone succeeded in %.1fs", elapsed)
            if not await disconnect_analysis_container(container, network):
                logger.warning(
                    "failed to isolate the container from the network — "
                    "failing closed for all remaining tools"
                )
                for tool in remaining:
                    _set_result(result, tool, _isolation_failed_evidence(tool, language))
                return result
            logger.info("analysis network removed — tools run with no network access")
            tool_budget = max(10, 90 - int(elapsed))
            outcomes = await _run_tools_parallel(
                container, language, _REPO_DIR, list(remaining), tool_budget
            )
            for tool, evidence in outcomes.items():
                cat = _classify_failure(
                    evidence.status, getattr(evidence, "error_message", None)
                )
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
            logger.info("transient clone failure — will re-clone in attempt 2")

        if not remaining:
            return result

        # ---------------- attempt 2 ----------------
        if await is_workspace_available(container):
            # Attempt 1 cloned successfully; only tool runs were transient.
            # Reuse the SAME container and its clone — no re-clone.
            logger.info(
                "attempt 2: reusing container clone for %d tool(s) with 180s budget",
                len(remaining),
            )
            tool_budget = 180
        else:
            # No usable clone in the container: re-clone inside the SAME
            # container with the 180s budget. The container is still on the
            # dedicated network, so the clone has network access again; it
            # is disconnected once more before the retry runs.
            logger.info(
                "attempt 2: re-cloning in the same container with 180s budget"
            )
            start = time.monotonic()
            clone_result = await clone_in_container(
                container, repo_url, pat, timeout_seconds=180
            )
            elapsed = time.monotonic() - start
            if clone_result.status != "success":
                logger.warning("attempt 2: re-clone failed — exhausted for all tools")
                for tool in remaining:
                    _set_result(result, tool, _exhausted_evidence(tool, language))
                return result
            if not await disconnect_analysis_container(container, network):
                logger.warning(
                    "attempt 2: failed to isolate the container from the network — "
                    "failing closed for all remaining tools"
                )
                for tool in remaining:
                    _set_result(result, tool, _isolation_failed_evidence(tool, language))
                return result
            tool_budget = max(10, 180 - int(elapsed))

        outcomes = await _run_tools_parallel(
            container, language, _REPO_DIR, list(remaining), tool_budget
        )
        for tool, evidence in outcomes.items():
            if evidence.status == "success":
                logger.info("attempt 2: %s succeeded", tool)
                _set_result(result, tool, evidence)
            else:
                logger.warning(
                    "attempt 2: %s status=%s — exhausted",
                    tool, evidence.status,
                )
                _set_result(result, tool, _exhausted_evidence(tool, language))
        return result
    finally:
        remove_analysis_container(container, network)