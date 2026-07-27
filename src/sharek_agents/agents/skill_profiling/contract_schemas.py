from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class ContractModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class RepositoryAuthorship(ContractModel):
    github_login: str = Field(min_length=1)
    repository_owned: bool
    recent_commit_count: int = Field(ge=0)
    total_commits: int = Field(ge=0)
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    contribution_detected: bool
    matched_recent_commit_shas: list[str]


class DependencyFileRef(ContractModel):
    filename: str = Field(min_length=1)
    parser_used: str | None = None


class FrameworkDetectionEvidence(ContractModel):
    frameworks_detected: dict[str, list[str]] = Field(default_factory=dict)
    dependency_files_identified: list[DependencyFileRef] = Field(default_factory=list)
    frameworks_count: int = Field(ge=0)
    status: Literal["success", "no_dependency_files", "parse_error"]


class CloneGateDecision(ContractModel):
    detected_framework_count: int = Field(ge=0)
    cloning_required: bool
    cloning_executed: bool
    reason: str | None = Field(default=None, max_length=500)
    pipeline_step_label: Literal["framework_detection_gate"] = "framework_detection_gate"


class StaticAnalysisEvidence(ContractModel):
    tool_used: str | None = None
    maintainability_index: float | None = None
    avg_cyclomatic_complexity: float | None = None
    lint_score: float | None = None
    lint_error_count: int | None = None
    files_evaluated_count: int | None = None
    status: Literal[
        "success", "language_not_supported", "no_analyzable_content",
        "repo_too_large", "tool_unavailable", "transient_failure_exhausted",
    ]


class GraphRelationsEvidence(ContractModel):
    inherits_count: int | None = None
    circular_imports_detected: bool | None = None
    coupling_summary: str | None = Field(default=None, max_length=500)
    status: Literal[
        "success", "no_analyzable_content", "repo_too_large",
        "tool_unavailable", "transient_failure_exhausted",
    ]


class RepositoryEvidenceCapsule(ContractModel):
    evidence_id: str = Field(min_length=1)
    full_name: str = Field(min_length=1)
    html_url: str = Field(min_length=1)
    private: bool
    fork: bool
    archived: bool
    default_branch: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    description: str | None
    topics: list[str]
    primary_language: str | None
    languages: dict[str, int]
    technologies: list[str]
    statistics: dict[str, Any]
    readme_excerpt: str | None
    contribution_activity: dict[str, Any]
    commit_signals: dict[str, Any]
    authorship: RepositoryAuthorship
    evidence_failures: list[str]
    static_analysis: StaticAnalysisEvidence | None = None
    graph_relations: GraphRelationsEvidence | None = None
    framework_detection: FrameworkDetectionEvidence | None = None
    clone_gate: CloneGateDecision | None = None

    @field_validator("languages")
    @classmethod
    def languages_must_be_non_negative(cls, value: dict[str, int]) -> dict[str, int]:
        if any(byte_count < 0 for byte_count in value.values()):
            raise ValueError("language byte counts must be non-negative")
        return value


class SkillProfileInput(ContractModel):
    contributor_id: str = Field(min_length=1)
    github_login: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    requested_at: datetime
    role: Literal["contributor", "owner"]
    # This PAT is request-scoped only. It must never be persisted to any
    # database or cache, and its lifetime is bounded to a single clone
    # operation performed by the analysis service. It is not a long-lived
    # credential and is not tied to any OAuth token's own expiry.
    github_pat: str | None = None
    selected_repositories: list[RepositoryEvidenceCapsule] = Field(
        min_length=1,
        max_length=10,
    )

    @field_validator("selected_repositories")
    @classmethod
    def evidence_ids_must_be_unique(
        cls, value: list[RepositoryEvidenceCapsule]
    ) -> list[RepositoryEvidenceCapsule]:
        evidence_ids = [repository.evidence_id for repository in value]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("selected repository evidence IDs must be unique")
        return value

    @field_validator("github_pat")
    @classmethod
    def github_pat_must_be_reasonable(
        cls, value: str | None
    ) -> str | None:
        if value is not None and (len(value) == 0 or len(value) > 255):
            raise ValueError(
                "github_pat must be a non-empty string of at most 255 characters"
            )
        return value


class GeneratedSkillCandidate(ContractModel):
    name: str = Field(min_length=1, max_length=100)
    proficiency: Literal["beginner", "intermediate", "advanced"]
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(min_length=1)
    evidence_summary: str | None = None
    limitations: list[str] = Field(default_factory=list)

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class FraudSignal(ContractModel):
    code: str = Field(min_length=1)
    severity: Literal["low", "medium", "high"]
    message: str = Field(min_length=1)
    repository_full_name: str | None = None


class ModelSkillProfileAnalysis(ContractModel):
    skills: list[GeneratedSkillCandidate]
    fraud_signals: list[FraudSignal] = Field(default_factory=list)


class SkillProfileResult(ModelSkillProfileAnalysis):
    evidence_quality: Literal["strong", "medium", "weak"]
    recommendation: Literal["pending_review", "needs_more_evidence"]
    provider: str = Field(min_length=1, max_length=50)
    model: str = Field(min_length=1, max_length=100)
    prompt_version: str = Field(min_length=1, max_length=100)
    schema_version: str = Field(min_length=1, max_length=100)
    service_version: str = Field(min_length=1, max_length=100)
