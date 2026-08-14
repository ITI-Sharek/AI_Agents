"""Deterministic evidence context for the Profiling LLM (Phase 29).

The Agent's Context Store (Phase 27) collects the tool outputs of one
request, and the graph summary layer (Phase 28) keeps large Graphify
outputs as-is while additionally storing a compact deterministic
summary. This module is the boundary between the two responsibilities:

* the Agent/Context Store collects evidence — nothing here retrieves,
  requests, or invents tool outputs,
* the Profiling LLM interprets the evidence — nothing here asks the LLM
  which outputs to retrieve or whether a graph is large.

The Profiling LLM receives exactly ONE explicit, deterministic evidence
package built from the stored context. Only evidence relevant to the
analysis mode is collected:

* CONTRIBUTOR: ``request``, ``technologies``, ``static_analysis``,
  ``full_graph`` (or ``full_graph_summary``), ``contributor_graph`` (or
  ``contributor_graph_summary``) — the contributor graph is the
  contributor-scoped evidence and the full graph is the
  repository-wide evidence,
* PROJECT: ``request``, ``technologies``, ``static_analysis``,
  ``full_graph`` (or ``full_graph_summary``) — contributor evidence
  never appears.

Graph representation rule (per graph, fully deterministic):

* when a summary is stored, the summary is the representation,
* otherwise the full graph is the representation.

The package never contains both representations of the same graph, and
the original full graph always remains stored in the Context Store
under its own key — this module never writes to the store.

The analysis-mode rule is the existing Phase 25/26 rule: CONTRIBUTOR
when the request carries a GitHub login, PROJECT otherwise.
"""

from __future__ import annotations

import json
from typing import Any

from sharek_agents.agents.skill_profiling.contract_schemas import SkillProfileInput
from sharek_agents.agents.skill_profiling_agent.context_store import (
    CONTEXT_CONTRIBUTOR_GRAPH,
    CONTEXT_CONTRIBUTOR_GRAPH_SUMMARY,
    CONTEXT_FULL_GRAPH,
    CONTEXT_FULL_GRAPH_SUMMARY,
    CONTEXT_REQUEST,
    CONTEXT_STATIC_ANALYSIS,
    CONTEXT_TECHNOLOGIES,
    ContextStore,
)

ANALYSIS_MODE_CONTRIBUTOR = "CONTRIBUTOR"
ANALYSIS_MODE_PROJECT = "PROJECT"

# Package keys by mode. The graph package keys keep the graph names
# (``full_graph`` / ``contributor_graph``); their VALUES follow the
# representation rule below.
_CONTRIBUTOR_PACKAGE_KEYS = (
    CONTEXT_REQUEST,
    CONTEXT_TECHNOLOGIES,
    CONTEXT_STATIC_ANALYSIS,
    CONTEXT_FULL_GRAPH,
    CONTEXT_CONTRIBUTOR_GRAPH,
)
_PROJECT_PACKAGE_KEYS = (
    CONTEXT_REQUEST,
    CONTEXT_TECHNOLOGIES,
    CONTEXT_STATIC_ANALYSIS,
    CONTEXT_FULL_GRAPH,
)

_GRAPH_SUMMARY_KEY = {
    CONTEXT_FULL_GRAPH: CONTEXT_FULL_GRAPH_SUMMARY,
    CONTEXT_CONTRIBUTOR_GRAPH: CONTEXT_CONTRIBUTOR_GRAPH_SUMMARY,
}


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
    """Package keys relevant to an analysis mode.

    CONTRIBUTOR includes the contributor graph; PROJECT never does.
    """
    if analysis_mode == ANALYSIS_MODE_CONTRIBUTOR:
        return _CONTRIBUTOR_PACKAGE_KEYS
    return _PROJECT_PACKAGE_KEYS


def _select_representation(
    context: ContextStore,
    graph_key: str,
    summary_key: str,
) -> str | None:
    """Determine the Profiling LLM representation of one stored graph.

    The summary is used when it exists; otherwise the full graph is
    used. Both are never returned, and the stored full graph is never
    modified.
    """
    summary = context.get(summary_key)
    if summary is not None:
        return summary
    return context.get(graph_key)


def build_evidence_context(
    context: ContextStore,
    analysis_mode: str,
) -> dict[str, Any]:
    """Build the explicit, deterministic evidence package for the Profiling LLM.

    The package always carries ``analysis_mode`` and only the evidence
    actually stored for that mode. Each graph is represented by its
    summary when one exists, otherwise by the full graph — never both.
    """
    package: dict[str, Any] = {"analysis_mode": analysis_mode}
    for key in evidence_keys_for_mode(analysis_mode):
        summary_key = _GRAPH_SUMMARY_KEY.get(key)
        if summary_key is not None:
            value = _select_representation(context, key, summary_key)
        else:
            value = context.get(key)
        if value is not None:
            package[key] = value
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