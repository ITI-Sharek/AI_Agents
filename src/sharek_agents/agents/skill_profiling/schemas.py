from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ErrorInfo(BaseModel):
    code: str
    message: str
    retryable: bool


class FrameworkSkill(BaseModel):
    name: str
    level: Literal["Beginner", "Mid-level", "Advanced", "Expert"]
    confidence: float
    evidence_type: Literal["github_stats", "static_analysis", "graphify_relations"]
    evidence: str


class GeneralSkill(BaseModel):
    name: Literal[
        "Clean Code",
        "Software Design & Architecture",
        "Code Quality & Maintainability",
        "Implementation",
        "Testing Practices",
    ]
    level: Literal["Beginner", "Mid-Level", "Advanced", "Expert"]
    confidence: float
    explanation: str = Field(description="Human-readable explanation of the level assignment")
    evidence: list[str] = Field(description="Concrete evidence items that justify the level")


class AgentResponse(BaseModel):
    status: Literal["success", "failed"]
    clean_code: GeneralSkill | None = None
    software_design_architecture: GeneralSkill | None = None
    code_quality_maintainability: GeneralSkill | None = None
    implementation: GeneralSkill | None = None
    testing_practices: GeneralSkill | None = None
    framework_skills: list[FrameworkSkill] = Field(default_factory=list)
    confidence: float | None = None
    sources: list[Source] = Field(default_factory=list)
    unresolved_repos: list[dict] = Field(default_factory=list)
    error: ErrorInfo | None = None


class SkillProfilingResult(BaseModel):
    clean_code: GeneralSkill
    software_design_architecture: GeneralSkill
    code_quality_maintainability: GeneralSkill
    implementation: GeneralSkill
    testing_practices: GeneralSkill
    framework_skills: list[FrameworkSkill] = Field(
        default_factory=list,
        description="Dynamic framework/library skill entries detected across repos",
    )


class Source(BaseModel):
    detail: str = Field(description="Human-readable detail with concrete numbers from the scoped evidence")
