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
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Exact evidence_id values that support this skill",
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


class RepositoryEvidenceCapsule(BaseModel):
    evidenceId: str = Field(min_length=1, max_length=250)
    fullName: str = Field(min_length=3, max_length=200)
    htmlUrl: str = Field(min_length=1, max_length=500)
    private: bool
    fork: bool
    archived: bool
    defaultBranch: str = Field(min_length=1, max_length=250)
    owner: str = Field(min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=1000)
    topics: list[str] = Field(default_factory=list, max_length=50)
    primaryLanguage: str | None = None
    languages: dict[str, int] = Field(default_factory=dict)
    technologies: list[str] = Field(default_factory=list, max_length=100)
    statistics: dict = Field(default_factory=dict)
    readmeExcerpt: str | None = Field(default=None, max_length=4000)
    contributionActivity: dict = Field(default_factory=dict)
    commitSignals: dict = Field(default_factory=dict)
    authorship: RepositoryAuthorship
    evidenceFailures: list[str] = Field(default_factory=list, max_length=20)


class RepositoryAuthorship(BaseModel):
    githubLogin: str = Field(min_length=1, max_length=100)
    repositoryOwned: bool
    recentCommitCount: int = Field(ge=0)
    totalCommits: int = Field(ge=0)
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    contributionDetected: bool
    matchedRecentCommitShas: list[str] = Field(default_factory=list, max_length=100)


class SkillProfileGenerationRequest(BaseModel):
    contributorId: str = Field(min_length=1, max_length=100)
    githubLogin: str = Field(min_length=1, max_length=100)
    generationId: str = Field(min_length=1, max_length=100)
    selectedRepositories: list[RepositoryEvidenceCapsule] = Field(
        min_length=1,
        max_length=10,
    )
    requestedAt: str = Field(min_length=1, max_length=100)


class GeneratedSkillCandidate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    proficiency: Literal["beginner", "intermediate", "advanced"]
    confidence: float = Field(ge=0, le=1)
    evidenceIds: list[str] = Field(min_length=1, max_length=10)
    evidenceSummary: str | None = Field(default=None, max_length=2000)
    limitations: list[str] = Field(default_factory=list, max_length=20)


class FraudSignal(BaseModel):
    code: str
    severity: Literal["low", "medium", "high"]
    message: str
    repositoryFullName: str | None = None


class SkillProfileGenerationResponse(BaseModel):
    skills: list[GeneratedSkillCandidate]
    fraudSignals: list[FraudSignal] = Field(default_factory=list)
    evidenceQuality: Literal["strong", "medium", "weak"]
    recommendation: Literal["pending_review", "needs_more_evidence"]
    provider: str
    model: str
    promptVersion: str
    schemaVersion: str
    serviceVersion: str
