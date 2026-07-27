from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ErrorInfo(BaseModel):
    code: str
    message: str
    retryable: bool


class Skill(BaseModel):
    name: str
    level: Literal["Beginner", "Mid-level", "Advanced", "Expert"]
    confidence: float
    evidence_type: Literal["github_stats", "static_analysis", "graphify_relations"]
    evidence: str


class AgentResponse(BaseModel):
    status: Literal["success", "failed"]
    skills: list[Skill] = Field(default_factory=list)
    confidence: float | None = None
    sources: list[Source] = Field(default_factory=list)
    unresolved_repos: list[dict] = Field(default_factory=list)
    error: ErrorInfo | None = None


class SkillProfilingResult(BaseModel):
    skills: list[Skill] = Field(
        default_factory=list,
        description="Unified list of all detected skills — frameworks, libraries, and general engineering practices",
    )


class Source(BaseModel):
    detail: str = Field(description="Human-readable detail with concrete numbers from the scoped evidence")
