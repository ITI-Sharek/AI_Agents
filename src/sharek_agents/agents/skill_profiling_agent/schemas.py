"""Agent-internal schemas for the new Skill Profiling Agent.

The request side reuses the existing wire contract
(``SkillProfileInput`` from ``skill_profiling.contract_schemas``). The
response below is the ReAct agent response: the final answer plus tool
activity records for debugging, and — since Phase 15 — a structured
skill profile built from validated evidence.

Since the contract alignment, the public skill object exposes exactly
``name``, ``level`` (beginner | intermediate | advanced | expert),
``confidence`` (0..1) and a human-readable ``evidence`` string. The
agent's own ``AgentSkill`` replaces the legacy contract's
``GeneratedSkillCandidate`` in this response; the legacy schema is
untouched and still serves ``/skill-profiles/generate``. Evidence IDs
remain for traceability as ``evidenceIds``, and limitations are kept.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from sharek_agents.agents.skill_profiling.contract_schemas import ContractModel


class AgentSkill(ContractModel):
    """One skill in the public agent response (contract-aligned).

    Mandatory concepts: ``name``, ``level``, ``confidence``, ``evidence``.
    ``evidence_ids`` stays for traceability (serialized as
    ``evidenceIds``); ``limitations`` is preserved. Raw internal
    ``EvidenceBundle`` records are never exposed here.
    """

    name: str = Field(min_length=1, max_length=100)
    level: Literal["beginner", "intermediate", "advanced", "expert"]
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(
        min_length=1,
        description="Human-readable summary of the evidence supporting the claim",
    )
    evidence_ids: list[str] = Field(
        min_length=1,
        description="Evidence IDs cited by this skill, for traceability",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Honest caveats about what the evidence does not prove",
    )


class ToolActivity(BaseModel):
    """Public record of one tool execution (debugging only).

    Intentionally contains no chain-of-thought or private reasoning.
    Arguments are never included; only the tool name, outcome, and a
    truncated safe result are exposed.
    """

    tool: str = Field(description="Tool name that was executed")
    status: Literal["success", "error"] = Field(description="Execution outcome")
    result_summary: str = Field(
        default="",
        description="Truncated, safe result that was returned to the agent",
    )
    error_message: str | None = Field(
        default=None,
        description="Safe error description, present when the execution failed",
    )


class SkillProfileAgentOutput(BaseModel):
    """Structured skill profile derived from the run's validated evidence.

    ``skills`` uses the agent's public ``AgentSkill`` shape (name, level,
    confidence, evidence). ``insufficient_evidence`` and
    ``recommendation`` follow the contract's semantics: when not enough
    valid evidence exists, no skills are fabricated and the profile
    reports ``needs_more_evidence``.
    """

    skills: list[AgentSkill] = Field(
        default_factory=list,
        description="Validated skills, at most 20, each citing real evidence",
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="True when no trustworthy skills could be produced",
    )
    recommendation: Literal["pending_review", "needs_more_evidence"] = Field(
        default="pending_review",
        description="Contract-style review recommendation",
    )
    message: str = Field(
        default="",
        description="Profile-level status or deterministic validation note",
    )


class SkillProfileAgentResponse(BaseModel):
    """Response of the Skill Profiling Agent (ReAct core, Phase 2).

    Carries the agent's final answer plus execution metadata and the
    structured skill profile (Phase 15). Failures such as provider
    errors or timeouts are surfaced as HTTP errors by the endpoint, not
    inside this schema.
    """

    status: Literal["success"] = "success"
    agent: Literal["skill_profiling_agent"] = "skill_profiling_agent"
    phase: Literal["react_core"] = "react_core"
    generation_id: str = Field(
        description="Echo of the request generation ID",
    )
    contributor_id: str = Field(
        description="Echo of the request contributor ID",
    )
    selected_repository_count: int = Field(
        ge=0,
        description="Number of selected repository evidence capsules received",
    )
    iterations_used: int = Field(
        default=0,
        ge=0,
        description="Number of ReAct loop iterations executed",
    )
    tool_activities: list[ToolActivity] = Field(
        default_factory=list,
        description="Tool executions performed during the run, for debugging",
    )
    message: str = Field(
        default="",
        description="The agent's final answer, or a terminal status note",
    )
    skill_profile: SkillProfileAgentOutput | None = Field(
        default=None,
        description=(
            "Structured skill profile derived from the evidence collected "
            "during this run; None when the run produced no profile"
        ),
    )
