"""Request-scoped evidence bundle for the ReAct Skill Profiling Agent (Phase 15).

Every meaningful tool result that can support a skill claim receives a
deterministic evidence ID and is stored in an ``EvidenceBundle`` scoped
to one agent run. The bundle also pre-registers the request's repository
evidence capsules using their existing contract ``evidence_id`` values,
so citations stay compatible with the legacy contract semantics.

Rules enforced here:

* only ``success`` tool results with non-empty output become evidence,
* tool failures (``execution_error``, ``validation_error``, ``not_found``,
  ``empty``) never become evidence,
* evidence IDs are derived deterministically from the tool name and its
  serialized arguments — never random,
* duplicate evidence IDs are deduplicated (identical tool calls collapse
  into one record),
* records never contain secrets: arguments are hashed, never stored, and
  only the already-safe tool output is preserved.

The bundle is request-scoped and never shared between runs; it is never
serialized into the agent response.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from sharek_agents.agents.skill_profiling.contract_schemas import (
    RepositoryEvidenceCapsule,
    SkillProfileInput,
)
from sharek_agents.agents.skill_profiling_agent.tools import ToolResult

logger = logging.getLogger(__name__)

EVIDENCE_ID_PREFIX = "ev:"
RESULT_SUMMARY_LIMIT = 500

# Tool-name -> evidence type. Unknown tools (e.g. dynamically discovered
# MCP tools) fall back to ``tool_result``.
_EVIDENCE_TYPE_BY_TOOL = {
    "detect_frameworks": "framework_detection",
    "get_agent_context": "repository_context",
    "acquire_repository": "repository_acquisition",
    "filter_contributor_code": "ownership",
    "analyze_static": "static_analysis",
    "analyze_graph": "graph_relations",
}

_REPOSITORY_ARGUMENT_KEYS = (
    "repository",
    "repository_identifier",
    "repo_url",
    "repo",
)

_CONTRIBUTOR_ARGUMENT_KEYS = (
    "contributor_identifier",
    "contributor",
    "contributor_id",
)


class EvidenceRecord(BaseModel):
    """One unit of evidence collected during a single agent run.

    Preserves enough information to identify the source tool, the
    repository/contributor scope when it is derivable, the evidence
    type, and the actual (safe) tool result.
    """

    evidence_id: str = Field(
        description="Deterministic, run-scoped evidence identifier"
    )
    source_tool: str = Field(description="Tool that produced the evidence")
    evidence_type: str = Field(description="Kind of evidence, e.g. framework_detection")
    scope: Literal["repository", "contributor", "request"] = Field(
        default="request",
        description="Scope the evidence applies to",
    )
    repository: str | None = Field(
        default=None,
        description="Repository full name (owner/name) when derivable from the call",
    )
    contributor: str | None = Field(
        default=None,
        description="Contributor identifier when derivable from the call",
    )
    result_summary: str = Field(
        default="",
        description="Truncated, safe summary of the tool result",
    )
    result: str = Field(
        default="",
        description="Full safe tool result that was returned to the agent",
    )


def _truncate(text: str, limit: int = RESULT_SUMMARY_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def _normalize_repository(value: str) -> str:
    """Reduce a repository identifier to its ``owner/name`` form.

    Handles ``owner/name``, ``https://github.com/owner/name[.git]`` and
    ``git@github.com:owner/name[.git]`` shapes. Anything unrecognized is
    returned unchanged.
    """
    candidate = value.strip()
    for prefix in (
        "https://github.com/",
        "http://github.com/",
        "git@github.com:",
        "github.com/",
    ):
        candidate = candidate.removeprefix(prefix)
    candidate = candidate.removesuffix(".git")
    parts = [part for part in candidate.split("/") if part]
    if len(parts) == 2:
        return "/".join(parts)
    return value.strip()


def _extract_repository(arguments: dict[str, Any]) -> str | None:
    for key in _REPOSITORY_ARGUMENT_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_repository(value)
    return None


def _extract_contributor(arguments: dict[str, Any]) -> str | None:
    for key in _CONTRIBUTOR_ARGUMENT_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _capsule_summary(capsule: RepositoryEvidenceCapsule) -> str:
    """Compact, safe serialization of one repository evidence capsule."""
    payload = {
        "evidence_id": capsule.evidence_id,
        "full_name": capsule.full_name,
        "default_branch": capsule.default_branch,
        "primary_language": capsule.primary_language,
        "languages": capsule.languages,
        "technologies": capsule.technologies,
        "topics": capsule.topics,
        "description": capsule.description,
        "authorship": {
            "repository_owned": capsule.authorship.repository_owned,
            "contribution_detected": capsule.authorship.contribution_detected,
            "recent_commit_count": capsule.authorship.recent_commit_count,
            "total_commits": capsule.authorship.total_commits,
            "additions": capsule.authorship.additions,
            "deletions": capsule.authorship.deletions,
        },
    }
    return json.dumps(payload, ensure_ascii=False)


class EvidenceBundle:
    """Request-scoped collection of evidence for one agent run.

    Pre-registers the request's repository evidence capsules (using the
    contract's exact ``evidence_id`` values) and records successful tool
    results under deterministic IDs. Failures and empty results are
    ignored; duplicate IDs collapse into the first record.
    """

    def __init__(self, request: SkillProfileInput) -> None:
        self._records: dict[str, EvidenceRecord] = {}
        self._order: list[str] = []
        self._pre_register(request)

    @staticmethod
    def derive_id(tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        """Derive a deterministic evidence ID for one tool call.

        The ID is a one-way hash of the tool name and its serialized
        arguments (sorted keys), so identical calls always produce the
        same ID and arguments never leak into the ID.
        """
        serialized = json.dumps(
            arguments or {},
            sort_keys=True,
            ensure_ascii=False,
        )
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
        return f"{EVIDENCE_ID_PREFIX}{tool_name}:{digest}"

    def record(
        self,
        result: ToolResult,
        arguments: dict[str, Any] | None = None,
    ) -> EvidenceRecord | None:
        """Record one tool result as evidence when it is meaningful.

        Returns the stored record, or ``None`` when the result is a
        failure or empty — those never become evidence.
        """
        if result.status != "success":
            return None
        output = (result.output or "").strip()
        if not output:
            return None

        args = arguments or {}
        evidence_id = self.derive_id(result.name, args)
        existing = self._records.get(evidence_id)
        if existing is not None:
            return existing

        scope, repository, contributor = _scope_for(args)
        record = EvidenceRecord(
            evidence_id=evidence_id,
            source_tool=result.name,
            evidence_type=_EVIDENCE_TYPE_BY_TOOL.get(result.name, "tool_result"),
            scope=scope,
            repository=repository,
            contributor=contributor,
            result_summary=_truncate(output),
            result=output,
        )
        self._records[evidence_id] = record
        self._order.append(evidence_id)
        return record

    def contains(self, evidence_id: str) -> bool:
        return evidence_id in self._records

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        return self._records.get(evidence_id)

    def ids(self) -> list[str]:
        """Deterministic (insertion-ordered) list of collected IDs."""
        return list(self._order)

    def records(self) -> list[EvidenceRecord]:
        return [self._records[evidence_id] for evidence_id in self._order]

    @property
    def size(self) -> int:
        return len(self._records)

    def _pre_register(self, request: SkillProfileInput) -> None:
        """Register each request capsule under its contract evidence_id."""
        for capsule in request.selected_repositories:
            summary = _capsule_summary(capsule)
            record = EvidenceRecord(
                evidence_id=capsule.evidence_id,
                source_tool="request",
                evidence_type="repository_capsule",
                scope="repository",
                repository=capsule.full_name,
                result_summary=_truncate(summary),
                result=summary,
            )
            self._records[record.evidence_id] = record
            self._order.append(record.evidence_id)


def _scope_for(
    arguments: dict[str, Any],
) -> tuple[str, str | None, str | None]:
    """Derive evidence scope from the tool call arguments.

    Contributor scope wins when a contributor identifier is present,
    then repository scope, then request scope.
    """
    repository = _extract_repository(arguments)
    contributor = _extract_contributor(arguments)
    if contributor is not None:
        return "contributor", repository, contributor
    if repository is not None:
        return "repository", repository, None
    return "request", None, None
