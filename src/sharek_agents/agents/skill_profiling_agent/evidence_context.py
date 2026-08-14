"""Deterministic evidence context for the Profiling LLM (Phase 29).

The Agent's Context Store collects the tool outputs of one request,
keeps each repository's evidence separate, and stores each Graphify
output exactly as returned by the MCP — the actual graph payload
(``nodes``, ``edges``, and any Graphify metadata) is never replaced by
an aggregate metric-only summary. This module is the boundary between
the two responsibilities:

* the Agent/Context Store collects evidence — nothing here retrieves,
  requests, or invents tool outputs,
* the Profiling LLM interprets the evidence — nothing here asks the LLM
  which outputs to retrieve or whether a graph is large.

The Profiling LLM receives exactly ONE explicit, deterministic evidence
package built from the stored context. The package carries the request
context and the evidence SEPARATED PER REPOSITORY, so analyzing
multiple repositories never collapses to the last one:

    {
      "analysis_mode": "...",
      "request": "...",
      "repositories": [
        {
          "repository": "<owner/name>",
          "technologies": "...",
          "static_analysis": "...",
          "full_graph": "...",
          "contributor_graph": "..."      // CONTRIBUTOR mode only
        },
        ...
      ]
    }

Only evidence relevant to the analysis mode is collected:

* CONTRIBUTOR: each repository contributes ``technologies``,
  ``static_analysis``, ``full_graph`` and ``contributor_graph`` — the
  contributor graph is the contributor-scoped evidence and the full
  graph is the repository-wide evidence,
* PROJECT: each repository contributes ``technologies``,
  ``static_analysis`` and ``full_graph`` — contributor evidence never
  appears.

Graph representation rule (per graph, fully deterministic): each graph
is represented by its actual stored Graphify payload — the full graph
or the selected contributor subgraph exactly as returned by the MCP.
No summary representation exists, and nothing here writes to the store.

The analysis-mode rule is the existing Phase 25/26 rule: CONTRIBUTOR
when the request carries a GitHub login, PROJECT otherwise.
"""

from __future__ import annotations

import json
from typing import Any

from sharek_agents.agents.skill_profiling.contract_schemas import SkillProfileInput
from sharek_agents.agents.skill_profiling_agent.context_store import (
    CONTEXT_CONTRIBUTOR_GRAPH,
    CONTEXT_FULL_GRAPH,
    CONTEXT_REQUEST,
    CONTEXT_STATIC_ANALYSIS,
    CONTEXT_TECHNOLOGIES,
    ContextStore,
)

ANALYSIS_MODE_CONTRIBUTOR = "CONTRIBUTOR"
ANALYSIS_MODE_PROJECT = "PROJECT"

# Per-repository package keys by mode. The same keys are also used for
# the orchestration observation.
_CONTRIBUTOR_REPOSITORY_KEYS = (
    CONTEXT_TECHNOLOGIES,
    CONTEXT_STATIC_ANALYSIS,
    CONTEXT_FULL_GRAPH,
    CONTEXT_CONTRIBUTOR_GRAPH,
)
_PROJECT_REPOSITORY_KEYS = (
    CONTEXT_TECHNOLOGIES,
    CONTEXT_STATIC_ANALYSIS,
    CONTEXT_FULL_GRAPH,
)


def analysis_mode_for_request(request: SkillProfileInput) -> str:
    """Return the existing Phase 25/26 analysis mode for a request.

    CONTRIBUTOR when the request carries a GitHub login, PROJECT
    otherwise — the same rule the Agent and the MCP adapters already
    use.
    """
    if request.github_login.strip():
        return ANALYSIS_MODE_CONTRIBUTOR
    return ANALYSIS_MODE_PROJECT


def evidence_keys_for_mode(analysis_mode: str) -> tuple[str, ...]:
    """Package keys relevant to an analysis mode, per repository.

    CONTRIBUTOR includes the contributor graph; PROJECT never does.
    """
    if analysis_mode == ANALYSIS_MODE_CONTRIBUTOR:
        return _CONTRIBUTOR_REPOSITORY_KEYS
    return _PROJECT_REPOSITORY_KEYS


def build_evidence_context(
    context: ContextStore,
    analysis_mode: str,
) -> dict[str, Any]:
    """Build the explicit, deterministic evidence package for the Profiling LLM.

    The package always carries ``analysis_mode`` and, when stored, the
    request context plus the evidence SEPARATED PER REPOSITORY: every
    analyzed repository keeps its own ``technologies``,
    ``static_analysis`` and graphs, so one repository's evidence never
    overwrites another's. Each graph is represented by its actual
    stored Graphify payload — the graph is never replaced by a
    metric-only summary, and no summary representation exists.
    """
    package: dict[str, Any] = {"analysis_mode": analysis_mode}

    request = context.request_context()
    if request is not None:
        package[CONTEXT_REQUEST] = request

    repositories: list[dict[str, Any]] = []
    for repository, values in context.repositories().items():
        entry: dict[str, Any] = {"repository": repository}
        for key in evidence_keys_for_mode(analysis_mode):
            value = values.get(key)
            if value is not None:
                entry[key] = value
        repositories.append(entry)
    if repositories:
        package["repositories"] = repositories

    return package


def evidence_context_message(
    context: ContextStore,
    analysis_mode: str,
) -> str | None:
    """Serialize the deterministic evidence package for the LLM input.

    Returns ``None`` when no relevant evidence is stored for the mode,
    so an empty package is never sent to the Profiling LLM.
    """
    package = build_evidence_context(context, analysis_mode)
    if len(package) == 1:
        return None
    return json.dumps(package, ensure_ascii=False)


__all__ = [
    "ANALYSIS_MODE_CONTRIBUTOR",
    "ANALYSIS_MODE_PROJECT",
    "analysis_mode_for_request",
    "build_evidence_context",
    "evidence_context_message",
    "evidence_keys_for_mode",
]
