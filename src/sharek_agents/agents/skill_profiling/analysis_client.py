from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from sharek_agents.agents.skill_profiling.contract_schemas import (
    GraphRelationsEvidence,
    RepositoryEvidenceCapsule,
    StaticAnalysisEvidence,
)
from sharek_agents.config import settings

logger = logging.getLogger(__name__)

_STATIC_STATUS_MAP: dict[str, str] = {
    "success": "success",
    "language_not_supported": "language_not_supported",
    "no_analyzable_content": "no_analyzable_content",
    "repo_too_large": "repo_too_large",
    "tool_unavailable": "tool_unavailable",
    "transient_failure_exhausted": "transient_failure_exhausted",
}

_GRAPH_STATUS_MAP: dict[str, str] = {
    "success": "success",
    "no_analyzable_content": "no_analyzable_content",
    "repo_too_large": "repo_too_large",
    "tool_unavailable": "tool_unavailable",
    "transient_failure_exhausted": "transient_failure_exhausted",
}


def _map_static_analysis(api_data: dict) -> StaticAnalysisEvidence:
    api_status = api_data.get("status", "tool_unavailable")
    return StaticAnalysisEvidence(
        tool_used="code-analysis-engine",
        maintainability_index=api_data.get("maintainability_index"),
        avg_cyclomatic_complexity=api_data.get("complexity"),
        lint_score=None,
        lint_error_count=None,
        files_evaluated_count=api_data.get("files_analyzed"),
        status=_STATIC_STATUS_MAP.get(api_status, "tool_unavailable"),
    )


def _map_graph_relations(api_data: dict) -> GraphRelationsEvidence:
    api_status = api_data.get("status", "tool_unavailable")
    circular = api_data.get("circular_imports")
    return GraphRelationsEvidence(
        inherits_count=api_data.get("inheritance_depth"),
        circular_imports_detected=bool(circular) if circular is not None else None,
        coupling_summary=api_data.get("coupling_summary"),
        status=_GRAPH_STATUS_MAP.get(api_status, "tool_unavailable"),
    )


def _set_capsule_evidence(
    repo: RepositoryEvidenceCapsule,
    parsed: dict[str, Any],
) -> None:
    sa_data = parsed.get("static_analysis")
    if sa_data is not None:
        try:
            repo.static_analysis = _map_static_analysis(sa_data)
        except Exception:
            logger.warning("Failed to map static_analysis for %s", repo.full_name)
            repo.static_analysis = StaticAnalysisEvidence(status="tool_unavailable")
    else:
        repo.static_analysis = StaticAnalysisEvidence(status="tool_unavailable")

    gr_data = parsed.get("graph_relations")
    if gr_data is not None:
        try:
            repo.graph_relations = _map_graph_relations(gr_data)
        except Exception:
            logger.warning("Failed to map graph_relations for %s", repo.full_name)
            repo.graph_relations = GraphRelationsEvidence(status="tool_unavailable")
    else:
        repo.graph_relations = GraphRelationsEvidence(status="tool_unavailable")


async def analyze_repo_for_capsule(
    repo: RepositoryEvidenceCapsule,
    pat: str | None,
    timeout: int = 190,
) -> None:
    """Call the REST analysis service once per repository.

    The *github_pat* is sent in the POST JSON body — it is never stored,
    cached, or logged on the orchestrator side.
    """
    language = repo.primary_language or "python"
    payload: dict[str, Any] = {
        "repo_url": repo.html_url,
        "language": language,
        "requested_tools": ["static_analysis", "graph_relations"],
        "pat": pat,
    }
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.analysis_service_auth_token:
        headers["Authorization"] = f"Bearer {settings.analysis_service_auth_token}"

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            resp = await client.post(
                f"{settings.analysis_service_url}/analyze/repo",
                json=payload,
                headers=headers,
            )
        resp.raise_for_status()
    except httpx.TimeoutException:
        logger.warning("Analysis service timed out for %s", repo.full_name)
        repo.static_analysis = StaticAnalysisEvidence(status="tool_unavailable")
        repo.graph_relations = GraphRelationsEvidence(status="tool_unavailable")
        return
    except httpx.HTTPStatusError:
        logger.warning("Analysis service returned HTTP error for %s", repo.full_name)
        repo.static_analysis = StaticAnalysisEvidence(status="tool_unavailable")
        repo.graph_relations = GraphRelationsEvidence(status="tool_unavailable")
        return
    except httpx.RequestError as exc:
        logger.warning("Analysis service unreachable for %s: %s", repo.full_name, exc)
        repo.static_analysis = StaticAnalysisEvidence(status="tool_unavailable")
        repo.graph_relations = GraphRelationsEvidence(status="tool_unavailable")
        return

    try:
        parsed = resp.json()
    except (json.JSONDecodeError, ValueError):
        logger.warning("Analysis service returned non-JSON for %s", repo.full_name)
        repo.static_analysis = StaticAnalysisEvidence(status="tool_unavailable")
        repo.graph_relations = GraphRelationsEvidence(status="tool_unavailable")
        return

    _set_capsule_evidence(repo, parsed)


async def run_step2_analysis(
    repos: list[RepositoryEvidenceCapsule],
    github_pat: str | None,
) -> None:
    """Step 2 — Run analysis for each repository via the analysis service REST API.

    Calls ``POST /analyze/repo`` once per repository, independently.
    HTTP-level failures (timeout, connection refused, 5xx) are handled
    per-repo — that repository's evidence gets a failure status and
    other repositories proceed unaffected.

    The *github_pat* is passed directly in the HTTP body for each
    per-repo call — never stored, cached, or logged. It is scoped to a
    single request and discarded when the response is processed.
    """
    if not repos:
        return

    if not settings.analysis_service_enabled:
        logger.info("Analysis service disabled — skipping Step 2")
        return

    for repo in repos:
        await analyze_repo_for_capsule(
            repo,
            github_pat,
            timeout=settings.analysis_service_timeout,
        )
