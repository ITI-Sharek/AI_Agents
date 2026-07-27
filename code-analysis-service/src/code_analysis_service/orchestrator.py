from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Literal

from .adapters import get_adapter
from .clone import clone_repo
from .graphify_runner import run_graphify
from .models import (
    AnalysisResult,
    EXTENSIONS,
    GraphRelationsEvidence,
    StaticAnalysisEvidence,
)

logger = logging.getLogger(__name__)

_DETERMINISTIC_STATUSES = frozenset({
    "language_not_supported",
    "no_analyzable_content",
    "repo_too_large",
    "tool_unavailable",
    "authentication_failed",
})


def _classify_failure(status: str, error_message: str | None = None) -> str:
    """Returns 'transient', 'deterministic', or 'success'."""
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


async def _attempt_clone(
    repo_url: str,
    pat: str | None,
    dest_dir: str,
    budget: int,
) -> tuple[bool, float]:
    start = time.monotonic()
    clone_result = await clone_repo(
        repo_url=repo_url,
        pat=pat,
        dest_dir=dest_dir,
        timeout_seconds=budget,
    )
    elapsed = time.monotonic() - start

    if clone_result.status == "success":
        logger.info("clone succeeded in %.1fs", elapsed)
        return True, elapsed

    cat = _classify_failure(clone_result.status, clone_result.error_message)
    logger.info(
        "clone status=%s category=%s elapsed=%.1fs",
        clone_result.status, cat, elapsed,
    )
    return False, elapsed


async def _run_static_analysis(
    language: str,
    repo_path: str,
    budget: int,
) -> StaticAnalysisEvidence:
    adapter = get_adapter(language)
    exts = EXTENSIONS.get(language.lower(), [])
    all_files = [
        str(p)
        for p in sorted(Path(repo_path).rglob("*"))
        if p.is_file() and p.suffix in exts
    ]
    if not all_files:
        return StaticAnalysisEvidence(
            status="no_analyzable_content", language=language
        )
    return adapter(repo_path=repo_path, file_paths=all_files, timeout=budget)


async def _run_tool(
    tool: str,
    language: str,
    repo_path: str,
    budget: int,
) -> StaticAnalysisEvidence | GraphRelationsEvidence:
    if tool == "static_analysis":
        return await _run_static_analysis(language, repo_path, budget)
    if tool == "graph_relations":
        return await run_graphify(
            cloned_repo_path=repo_path, timeout_seconds=budget
        )
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


async def analyze_repo(
    repo_url: str,
    pat: str | None,
    language: str,
    requested_tools: list[Literal["static_analysis", "graph_relations"]],
) -> AnalysisResult:
    result = AnalysisResult()
    remaining = set(requested_tools)

    # --- FIRST ATTEMPT: 90s budget ---
    first_dir = tempfile.mkdtemp(prefix="code-analysis-attempt1-")
    try:
        start = time.monotonic()
        clone_result = await clone_repo(
            repo_url=repo_url,
            pat=pat,
            dest_dir=first_dir,
            timeout_seconds=90,
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
                logger.info(
                    "deterministic clone failure — no retry for any tool"
                )
                for tool in remaining:
                    _set_result(
                        result, tool,
                        _clone_to_evidence(tool, language, clone_result.status),
                    )
                return result
            # transient clone failure -> tools will retry
            logger.info("transient clone failure — will retry all tools")
    finally:
        shutil.rmtree(first_dir, ignore_errors=True)

    if not remaining:
        return result

    # --- SECOND ATTEMPT: 180s budget (retry only) ---
    logger.info("retrying %d tool(s) with 180s budget", len(remaining))
    second_dir = tempfile.mkdtemp(prefix="code-analysis-attempt2-")
    try:
        clone_ok, elapsed = await _attempt_clone(repo_url, pat, second_dir, 180)

        for tool in list(remaining):
            if not clone_ok:
                logger.warning("retry: %s — clone failed, exhausted", tool)
                _set_result(result, tool, _exhausted_evidence(tool, language))
                continue

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