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

Evidence is stored PER REPOSITORY. The repository-scoped evidence
(``technologies``, ``static_analysis``, ``full_graph``,
``contributor_graph``) is keyed by the normalized repository identity
(owner/name — the same identity ``EvidenceRecord.repository`` uses), so
analyzing repository R2 never overwrites the stored evidence of R1:

    request             — ``get_agent_context`` output (request-level,
                           never per repository)
    repositories        — repo identity -> {context key -> stored output}

The repository identity is taken from existing tool metadata — the
orchestration envelope's ``repository.repo_url`` / ``repository.
repository_identifier`` and the tool call arguments (``repo_url`` /
``repository`` / ``repository_identifier``) — never an invented
identifier. All references normalize to the same identity.

Deterministic per-repository context keys:

  * ``technologies``       — ``detect_frameworks`` output
  * ``static_analysis``    — orchestration static analysis evidence
  * ``full_graph``         — Graphify output for the FULL repository
  * ``contributor_graph``  — Graph Select output for the contributor
                             scope (CONTRIBUTOR mode only)

The graph distinction is strict: ``full_graph`` and
``contributor_graph`` are stored under separate keys and one is never
overwritten by the other. PROJECT mode produces no contributor graph,
so no ``contributor_graph`` entry is ever created in that mode.

Graph outputs are always stored as-is, exactly as returned by the MCP:
the actual Graphify graph payload (``nodes``, ``edges``, and any
Graphify metadata) is preserved and is never replaced by an aggregate
metric-only summary representation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sharek_agents.agents.skill_profiling_agent.evidence import (
    _normalize_repository,
)

logger = logging.getLogger(__name__)

CONTEXT_REQUEST = "request"
CONTEXT_TECHNOLOGIES = "technologies"
CONTEXT_STATIC_ANALYSIS = "static_analysis"
CONTEXT_FULL_GRAPH = "full_graph"
CONTEXT_CONTRIBUTOR_GRAPH = "contributor_graph"

_ORCHESTRATION_TOOL = "analyze_contributor_repository"

# Repository-scoped context keys, in deterministic storage order.
_REPOSITORY_CONTEXT_KEYS = (
    CONTEXT_TECHNOLOGIES,
    CONTEXT_STATIC_ANALYSIS,
    CONTEXT_FULL_GRAPH,
    CONTEXT_CONTRIBUTOR_GRAPH,
)

# Existing argument keys that carry a repository reference (the same
# argument names the tools already declare).
_REPOSITORY_ARGUMENT_KEYS = (
    "repo_url",
    "repository_identifier",
    "repository",
    "repo",
)


def _repository_key_from_arguments(arguments: dict[str, Any] | None) -> str | None:
    """Normalized repository identity from existing tool call arguments.

    Uses the repository reference the tool call already carries
    (``repo_url`` for the orchestration tool, ``repository`` for
    ``detect_frameworks``) and reduces every shape (URL or owner/name)
    to the same normalized identity used by ``EvidenceRecord``.
    """
    if not arguments:
        return None
    for key in _REPOSITORY_ARGUMENT_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_repository(value)
    return None


class ContextStore:
    """Deterministic per-request store of meaningful tool outputs.

    One instance per Skill Profiling request; never global, never
    shared between requests, never persisted. Repository-scoped entries
    are keyed by the normalized repository identity so one
    repository's evidence can never overwrite another repository's
    evidence. Only successful tool results with non-empty output are
    stored, and only for tools that are part of the Skill Profiling
    evidence flow.

    Graphify outputs are stored exactly as returned by the MCP: the
    actual graph payload (``nodes``, ``edges``, and any Graphify
    metadata) is preserved — no aggregate metric-only summary
    representation is generated.
    """

    def __init__(self) -> None:
        self._request: str | None = None
        self._repositories: dict[str, dict[str, str]] = {}

    def record_tool_result(
        self,
        tool_name: str,
        output: str,
        arguments: dict[str, Any] | None = None,
    ) -> list[str]:
        """Store one successful tool output under deterministic key(s).

        Returns the context keys written (empty when the tool is not
        part of the evidence flow or the output is empty). This is
        called automatically by the agent execution layer for every
        tool result — the LLM is never asked whether an output should
        be stored. Repository-scoped outputs are stored under the
        repository identity taken from the existing tool metadata, so
        each repository keeps its own evidence.
        """
        if not output:
            return []
        if tool_name == "get_agent_context":
            self._request = output
            return [CONTEXT_REQUEST]
        if tool_name == "detect_frameworks":
            repository = _repository_key_from_arguments(arguments)
            if repository is None:
                return []
            self._repositories.setdefault(repository, {})[
                CONTEXT_TECHNOLOGIES
            ] = output
            return [CONTEXT_TECHNOLOGIES]
        if tool_name == _ORCHESTRATION_TOOL:
            return self._record_orchestration_output(output, arguments)
        return []

    def request_context(self) -> str | None:
        """Return the stored request context, or None when absent."""
        return self._request

    def repositories(self) -> dict[str, dict[str, str]]:
        """Return the stored evidence per repository (insertion-ordered).

        Mapping: normalized repository identity -> {context key ->
        stored output}. Each repository's values are independent:
        recording evidence for one repository never overwrites another
        repository's evidence. Callers must not mutate the mapping.
        """
        return self._repositories

    def repository_values(self, repository: str) -> dict[str, str] | None:
        """Return the stored evidence for one repository identity, or None."""
        return self._repositories.get(repository)

    def orchestration_repository(self, output: str) -> str | None:
        """Return the normalized repository identity of an orchestration envelope.

        Reads the existing repository metadata the MCP envelope already
        carries (``repository.repo_url`` — the canonical repository URL —
        and ``repository.repository_identifier``) and reduces it to the
        normalized identity. Never an invented identifier.
        """
        try:
            payload = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        repository = payload.get("repository")
        if isinstance(repository, dict):
            for key in ("repo_url", "repository_identifier"):
                value = repository.get(key)
                if isinstance(value, str) and value.strip():
                    return _normalize_repository(value)
        return None

    def _record_orchestration_output(
        self,
        output: str,
        arguments: dict[str, Any] | None,
    ) -> list[str]:
        """Store the orchestration result envelope under one repository.

        The repository identity comes from the envelope's own
        ``repository`` metadata (canonical URL preferred) with the call
        arguments as fallback. The ``analyze_contributor_repository``
        result envelope carries the ``static_analysis`` evidence and a
        ``graph`` section. The full-repository Graphify output
        (``graph.full_repository_graphify``) is stored under
        ``full_graph``; the contributor-scope Graph Select output
        (``graph.contributor_graph``) is stored under
        ``contributor_graph`` — only when it actually exists
        (CONTRIBUTOR mode), so the two graphs can never overwrite each
        other. Each graph section is stored as-is, exactly as produced
        by the MCP — the actual Graphify graph payload is preserved
        with no summary representation.
        """
        repository = self.orchestration_repository(output)
        if repository is None:
            repository = _repository_key_from_arguments(arguments)
        if repository is None:
            return []

        try:
            payload = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(payload, dict):
            return []

        keys: list[str] = []
        entry = self._repositories.setdefault(repository, {})

        static_analysis = payload.get("static_analysis")
        if isinstance(static_analysis, dict):
            entry[CONTEXT_STATIC_ANALYSIS] = json.dumps(
                static_analysis, ensure_ascii=False
            )
            keys.append(CONTEXT_STATIC_ANALYSIS)

        graph = payload.get("graph")
        if isinstance(graph, dict):
            full_graph = graph.get("full_repository_graphify")
            if isinstance(full_graph, dict):
                entry[CONTEXT_FULL_GRAPH] = json.dumps(
                    full_graph, ensure_ascii=False
                )
                keys.append(CONTEXT_FULL_GRAPH)
            contributor_graph = graph.get("contributor_graph")
            if isinstance(contributor_graph, dict):
                entry[CONTEXT_CONTRIBUTOR_GRAPH] = json.dumps(
                    contributor_graph, ensure_ascii=False
                )
                keys.append(CONTEXT_CONTRIBUTOR_GRAPH)
        return keys


__all__ = [
    "CONTEXT_CONTRIBUTOR_GRAPH",
    "CONTEXT_FULL_GRAPH",
    "CONTEXT_REQUEST",
    "CONTEXT_STATIC_ANALYSIS",
    "CONTEXT_TECHNOLOGIES",
    "ContextStore",
]
