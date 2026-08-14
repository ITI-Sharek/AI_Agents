"""Per-request in-memory Context Store for the Skill Profiling Agent (Phase 27).

Every ``generate_skill_profile_agent`` request gets ONE fresh
``ContextStore`` owned by the Agent run lifecycle. Meaningful tool
results are stored automatically by the agent execution layer as soon
as they return from the tool boundary — the LLM never decides what is
stored — and the stored outputs remain available to the Agent during
that same request.

This is intentionally NOT Redis, NOT a database, and NOT persistent
memory: no global mutable state, no module-level shared request state,
and concurrent requests never share a store.

Deterministic context keys:

  * ``request``            — ``get_agent_context`` output
  * ``technologies``       — ``detect_frameworks`` output
  * ``static_analysis``    — orchestration static analysis evidence
  * ``full_graph``         — Graphify output for the FULL repository
  * ``full_graph_summary`` — compact deterministic summary, ONLY when
                             the full graph exceeds the threshold
  * ``contributor_graph``  — Graph Select output for the contributor
                             scope (CONTRIBUTOR mode only)
  * ``contributor_graph_summary`` — compact deterministic summary,
                             ONLY when the contributor graph exceeds
                             the threshold

The graph distinction is strict: ``full_graph`` and
``contributor_graph`` are stored under separate keys and one is never
overwritten by the other. PROJECT mode produces no contributor graph,
so no ``contributor_graph`` entry is ever created in that mode.

Graph outputs are always stored as-is. Phase 28: when a graph exceeds
the configurable character threshold, a compact deterministic summary
(metrics and relation counts already present in the output — nothing
invented, no LLM involved) is stored ADDITIONALLY under the matching
``*_graph_summary`` key. The original graph is never overwritten or
deleted, and nothing is sent to the LLM in this phase.
"""

from __future__ import annotations

import json
import logging

from sharek_agents.agents.skill_profiling_agent.graph_summary import (
    DEFAULT_GRAPH_SUMMARY_THRESHOLD_CHARS,
    build_graph_summary,
    is_graph_too_large,
)

logger = logging.getLogger(__name__)

CONTEXT_REQUEST = "request"
CONTEXT_TECHNOLOGIES = "technologies"
CONTEXT_STATIC_ANALYSIS = "static_analysis"
CONTEXT_FULL_GRAPH = "full_graph"
CONTEXT_FULL_GRAPH_SUMMARY = "full_graph_summary"
CONTEXT_CONTRIBUTOR_GRAPH = "contributor_graph"
CONTEXT_CONTRIBUTOR_GRAPH_SUMMARY = "contributor_graph_summary"

_ORCHESTRATION_TOOL = "analyze_contributor_repository"

# Tool name -> context key for tools with a flat, single output.
_SINGLE_KEY_BY_TOOL = {
    "get_agent_context": CONTEXT_REQUEST,
    "detect_frameworks": CONTEXT_TECHNOLOGIES,
}


class ContextStore:
    """Deterministic per-request store of meaningful tool outputs.

    One instance per Skill Profiling request; never global, never
    shared between requests, never persisted. Entries are
    insertion-ordered and keyed by context type. Only successful tool
    results with non-empty output are stored, and only for tools that
    are part of the Skill Profiling evidence flow.

    Large Graphify outputs are additionally summarized deterministically
    (see ``skill_profiling_agent.graph_summary``): the original graph is
    always kept as-is and a compact summary is stored under a separate
    key only when the graph exceeds ``graph_summary_threshold_chars``.
    """

    def __init__(
        self,
        *,
        graph_summary_threshold_chars: int = DEFAULT_GRAPH_SUMMARY_THRESHOLD_CHARS,
    ) -> None:
        self._entries: dict[str, str] = {}
        self._graph_summary_threshold_chars = graph_summary_threshold_chars

    def record_tool_result(self, tool_name: str, output: str) -> list[str]:
        """Store one successful tool output under deterministic key(s).

        Returns the context keys written (empty when the tool is not
        part of the evidence flow or the output is empty). This is
        called automatically by the agent execution layer for every
        tool result — the LLM is never asked whether an output should
        be stored.
        """
        if not output:
            return []
        if tool_name == _ORCHESTRATION_TOOL:
            return self._record_orchestration_output(output)
        key = _SINGLE_KEY_BY_TOOL.get(tool_name)
        if key is None:
            return []
        self._entries[key] = output
        return [key]

    def get(self, key: str) -> str | None:
        """Return a previously stored output, or None when absent."""
        return self._entries.get(key)

    def contains(self, key: str) -> bool:
        """Whether a context entry with the given key exists."""
        return key in self._entries

    def keys(self) -> list[str]:
        """Deterministic (insertion-ordered) list of stored keys."""
        return list(self._entries)

    def size(self) -> int:
        """Number of stored entries."""
        return len(self._entries)

    def _record_orchestration_output(self, output: str) -> list[str]:
        """Split the orchestration result envelope into context keys.

        The ``analyze_contributor_repository`` result is a JSON envelope
        carrying the ``static_analysis`` evidence and a ``graph``
        section. The full-repository Graphify output
        (``graph.full_repository_graphify``) is stored under
        ``full_graph``; the contributor-scope Graph Select output
        (``graph.contributor_graph``) is stored under
        ``contributor_graph`` — only when it actually exists
        (CONTRIBUTOR mode), so the two graphs can never overwrite each
        other. Each section is stored as-is, exactly as produced by the
        tool; graphs exceeding the threshold additionally get a compact
        summary under their own key.
        """
        try:
            payload = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(payload, dict):
            return []

        keys: list[str] = []
        static_analysis = payload.get("static_analysis")
        if isinstance(static_analysis, dict):
            self._entries[CONTEXT_STATIC_ANALYSIS] = json.dumps(
                static_analysis, ensure_ascii=False
            )
            keys.append(CONTEXT_STATIC_ANALYSIS)

        graph = payload.get("graph")
        if isinstance(graph, dict):
            full_graph = graph.get("full_repository_graphify")
            if isinstance(full_graph, dict):
                keys.extend(
                    self._store_graph_section(
                        CONTEXT_FULL_GRAPH,
                        CONTEXT_FULL_GRAPH_SUMMARY,
                        full_graph,
                    )
                )
            contributor_graph = graph.get("contributor_graph")
            if isinstance(contributor_graph, dict):
                keys.extend(
                    self._store_graph_section(
                        CONTEXT_CONTRIBUTOR_GRAPH,
                        CONTEXT_CONTRIBUTOR_GRAPH_SUMMARY,
                        contributor_graph,
                    )
                )
        return keys

    def _store_graph_section(
        self,
        key: str,
        summary_key: str,
        graph: dict[str, object],
    ) -> list[str]:
        """Store one graph section as-is, summarizing only when large.

        The original graph output is always stored first and is never
        overwritten or deleted. When it exceeds the configurable
        threshold, a compact summary is stored as an ADDITIONAL
        representation under ``summary_key`` — never replacing the
        original. Returns the context keys written.
        """
        value = json.dumps(graph, ensure_ascii=False)
        self._entries[key] = value
        keys = [key]
        if is_graph_too_large(value, self._graph_summary_threshold_chars):
            summary = build_graph_summary(value)
            if summary is not None:
                self._entries[summary_key] = summary
                keys.append(summary_key)
        return keys


__all__ = [
    "CONTEXT_CONTRIBUTOR_GRAPH",
    "CONTEXT_CONTRIBUTOR_GRAPH_SUMMARY",
    "CONTEXT_FULL_GRAPH",
    "CONTEXT_FULL_GRAPH_SUMMARY",
    "CONTEXT_REQUEST",
    "CONTEXT_STATIC_ANALYSIS",
    "CONTEXT_TECHNOLOGIES",
    "ContextStore",
]