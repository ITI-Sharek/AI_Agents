from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    status: Literal["success", "failed"]
    data: SkillProfilingResult | None = None
    error_code: str | None = None
    retryable: bool | None = None


SkillEvidenceType = Literal["github_stats", "static_analysis", "graphify_relations"]


class RawSkill(BaseModel):
    name: str = Field(description="Name of the identified skill")
    evidence_type: SkillEvidenceType = Field(
        description="Which evidence category supports this skill"
    )
    description: str = Field(
        description="One-sentence description of what the developer demonstrated"
    )
    supporting_evidence: list[str] = Field(
        description="Concrete, numbers-backed evidence items that justify this skill"
    )


class SkillProfilingResult(BaseModel):
    skills: list[RawSkill] = Field(
        description="All skills identified from the developer's evidence"
    )
    overall_level: str = Field(
        description="Overall proficiency level: Beginner, Intermediate, or Advanced"
    )
    summary: str = Field(
        description="Two-to-three sentence summary of the developer's profile"
    )


class Source(BaseModel):
    type: SkillEvidenceType = Field(description="Evidence category")
    detail: str = Field(
        description="Human-readable detail with concrete numbers from the scoped evidence"
    )


class Skill(BaseModel):
    name: str = Field(description="Name of the identified skill")
    confidence: float = Field(
        description="Confidence score 0.0–1.0 based on strength and breadth of evidence"
    )
    sources: list[Source] = Field(
        description="Evidence sources that support this skill, each with scoped detail"
    )


class Contributor(BaseModel):
    username: str = Field(description="GitHub username")
    status: Literal["success", "needs_review"] = Field(
        description="Profile status — needs_review means insufficient or contradictory evidence"
    )
    confidence: float = Field(
        description="Overall confidence score 0.0–1.0 across all skills"
    )
    skills: list[Skill] = Field(
        description="Identified skills with per-skill confidence and source attribution"
    )
