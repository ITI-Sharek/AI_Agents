from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorInfo(BaseModel):
    code: str
    message: str
    retryable: bool


class AgentResponse(BaseModel):
    status: Literal["success", "failed"]
    data: Any = None
    general_skills: list[GeneralSkill] | None = None
    confidence: float | None = None
    sources: list[Source] = Field(default_factory=list)
    unresolved_repos: list[dict] = Field(default_factory=list)
    error: ErrorInfo | None = None


RepoEvidenceType = Literal["repo_metadata", "framework_detection", "static_analysis", "graphify_relations", "general_skill"]


class RawSkill(BaseModel):
    name: str = Field(description="Name of the identified skill")
    evidence_type: RepoEvidenceType = Field(
        description="Which evidence category supports this skill"
    )
    description: str = Field(
        description="One-sentence description of what the repository demonstrates"
    )
    supporting_evidence: list[str] = Field(
        description="Concrete, numbers-backed evidence items that justify this skill"
    )


class GeneralSkill(BaseModel):
    name: Literal[
        "Clean Code",
        "Software Design & Architecture",
        "Code Quality & Maintainability",
        "Implementation",
        "Testing Practices",
    ]
    level: Literal["Beginner", "Mid-level", "Advanced", "Expert"]
    confidence: float
    evidence: str


class SkillProfilingResult(BaseModel):
    skills: list[RawSkill] = Field(
        description="All skills identified from the repository evidence"
    )
    general_skills: list[GeneralSkill] = Field(
        description="Fixed-category general skill assessments: Clean Code, Software Design & Architecture, Code Quality & Maintainability, Implementation, Testing Practices"
    )
    overall_level: str = Field(
        description="Overall proficiency level: Beginner, Intermediate, or Advanced"
    )
    summary: str = Field(
        description="Two-to-three sentence summary of the repository profile"
    )


class Source(BaseModel):
    type: RepoEvidenceType = Field(description="Evidence category")
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


class RepoInfo(BaseModel):
    name: str = Field(description="Repository name")
    owner: str = Field(description="Repository owner")
    description: str = Field(description="Repository description")
    language: str = Field(description="Primary programming language")
    topics: list[str] = Field(description="Repository topics")
    stars: int = Field(description="Number of stars")
    forks: int = Field(description="Number of forks")
    clone_url: str = Field(description="Clone URL")
    default_branch: str = Field(description="Default branch name")


class RepoProfile(BaseModel):
    repo: RepoInfo = Field(description="Repository information")
    status: Literal["success", "needs_review"] = Field(
        description="Profile status — needs_review means insufficient or contradictory evidence"
    )
    confidence: float = Field(
        description="Overall confidence score 0.0–1.0 across all skills"
    )
    skills: list[Skill] = Field(
        description="Identified skills with per-skill confidence and source attribution"
    )



