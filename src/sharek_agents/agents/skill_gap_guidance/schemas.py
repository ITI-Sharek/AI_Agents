from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

GuidanceRequirementKind = Literal["required", "preferred"]
GuidanceProficiency = Literal["beginner", "intermediate", "advanced"]
GuidanceEvidenceType = Literal[
    "approved_skill",
    "contribution_requirement",
    "curated_learning_resource",
]
GuidanceGapKind = Literal["not_evidenced", "below_target_proficiency"]
GuidanceResourceType = Literal[
    "documentation",
    "course",
    "tutorial",
    "book",
    "reference",
]
GuidanceStatus = Literal[
    "COMPLETED",
    "NOT_STARTED_SYSTEM_LIMIT",
    "NOT_STARTED_NO_ASSESSABLE_EVIDENCE",
]


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class ContractModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class GuidanceRequirementSnapshot(ContractModel):
    id: str = Field(min_length=1, max_length=200)
    kind: GuidanceRequirementKind
    position: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=1000)

    @field_validator("id", "text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("requirement fields must not be blank")
        return stripped


class GuidanceApprovedSkillSnapshot(ContractModel):
    evidence_id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    proficiency: GuidanceProficiency
    evidence_summary: str | None = Field(default=None, max_length=2000)

    @field_validator("evidence_id", "name")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("approved skill fields must not be blank")
        return stripped


class GuidanceEvidenceCapsule(ContractModel):
    evidence_id: str = Field(min_length=1, max_length=200)
    type: GuidanceEvidenceType
    label: str = Field(min_length=1, max_length=300)
    summary: str | None = Field(default=None, max_length=2000)

    @field_validator("evidence_id", "label")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("evidence fields must not be blank")
        return stripped


class SkillGapGuidanceInput(ContractModel):
    guidance_request_id: str = Field(min_length=1, max_length=200)
    requirements: list[GuidanceRequirementSnapshot] = Field(
        default_factory=list, max_length=50
    )
    approved_skills: list[GuidanceApprovedSkillSnapshot] = Field(
        default_factory=list, max_length=50
    )
    evidence: list[GuidanceEvidenceCapsule] = Field(
        default_factory=list, max_length=100
    )
    allowed_evidence_ids: list[str] = Field(
        default_factory=list, max_length=100
    )
    requested_at: datetime
    contract_version: Literal["skill-gap-guidance-v1"] = "skill-gap-guidance-v1"

    @field_validator("guidance_request_id")
    @classmethod
    def request_id_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("guidance_request_id must not be blank")
        return stripped

    @field_validator("requirements")
    @classmethod
    def requirement_ids_must_be_unique(
        cls, value: list[GuidanceRequirementSnapshot]
    ) -> list[GuidanceRequirementSnapshot]:
        ids = [requirement.id for requirement in value]
        if len(ids) != len(set(ids)):
            raise ValueError("requirement snapshot IDs must be unique")
        return value

    @field_validator("approved_skills")
    @classmethod
    def skill_evidence_ids_must_be_unique(
        cls, value: list[GuidanceApprovedSkillSnapshot]
    ) -> list[GuidanceApprovedSkillSnapshot]:
        ids = [skill.evidence_id for skill in value]
        if len(ids) != len(set(ids)):
            raise ValueError("approved skill evidence IDs must be unique")
        return value

    @field_validator("evidence")
    @classmethod
    def evidence_ids_must_be_unique(
        cls, value: list[GuidanceEvidenceCapsule]
    ) -> list[GuidanceEvidenceCapsule]:
        ids = [item.evidence_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence IDs must be unique")
        return value

    @field_validator("allowed_evidence_ids")
    @classmethod
    def allowed_evidence_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("allowed evidence IDs must not be blank")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("allowed evidence IDs must be unique")
        return cleaned


class GuidanceMissingSkill(ContractModel):
    requirement_id: str = Field(min_length=1, max_length=200)
    skill_name: str = Field(min_length=1, max_length=200)
    gap: GuidanceGapKind
    explanation: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(min_length=1, max_length=50)
    uncertainty: list[str] = Field(default_factory=list, max_length=10)


class GuidanceRecommendedTechnology(ContractModel):
    name: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(min_length=1, max_length=50)


class GuidanceLearningResource(ContractModel):
    title: str = Field(min_length=1, max_length=300)
    resource_type: GuidanceResourceType
    url: str = Field(min_length=1, max_length=1000)
    rationale: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(min_length=1, max_length=50)

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith(("https://", "http://")):
            raise ValueError("learning resource URLs must use HTTP(S)")
        return value


class GuidancePracticeProject(ContractModel):
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=3000)
    technologies: list[str] = Field(default_factory=list, max_length=20)
    evidence_ids: list[str] = Field(min_length=1, max_length=50)


class GuidanceImprovementStep(ContractModel):
    step: str = Field(min_length=1, max_length=300)
    focus: str = Field(min_length=1, max_length=2000)
    estimated_duration: str | None = Field(default=None, max_length=100)
    evidence_ids: list[str] = Field(min_length=1, max_length=50)


class GuidanceSource(ContractModel):
    evidence_id: str = Field(min_length=1, max_length=200)
    label: str = Field(min_length=1, max_length=300)
    type: GuidanceEvidenceType


class GuidanceProviderOutput(ContractModel):
    missing_skills: list[GuidanceMissingSkill] = Field(default_factory=list)
    recommended_technologies: list[GuidanceRecommendedTechnology] = Field(
        default_factory=list
    )
    learning_resources: list[GuidanceLearningResource] = Field(
        default_factory=list
    )
    practice_projects: list[GuidancePracticeProject] = Field(
        default_factory=list
    )
    improvement_path: list[GuidanceImprovementStep] = Field(
        default_factory=list
    )
    sources: list[GuidanceSource] = Field(default_factory=list)


class GuidanceMetadata(ContractModel):
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=100)
    prompt_version: str = Field(min_length=1, max_length=100)
    schema_version: str = Field(min_length=1, max_length=100)
    service_version: str = Field(min_length=1, max_length=100)
    latency_ms: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class SkillGapGuidanceResult(ContractModel):
    status: GuidanceStatus = "COMPLETED"
    missing_skills: list[GuidanceMissingSkill] = Field(default_factory=list)
    recommended_technologies: list[GuidanceRecommendedTechnology] = Field(
        default_factory=list
    )
    learning_resources: list[GuidanceLearningResource] = Field(
        default_factory=list
    )
    practice_projects: list[GuidancePracticeProject] = Field(
        default_factory=list
    )
    improvement_path: list[GuidanceImprovementStep] = Field(
        default_factory=list
    )
    sources: list[GuidanceSource] = Field(default_factory=list)
    metadata: GuidanceMetadata | None = None

    @model_validator(mode="after")
    def payload_matches_status(self) -> SkillGapGuidanceResult:
        if self.status == "COMPLETED":
            if self.metadata is None:
                raise ValueError("completed guidance requires metadata")
            return self

        if any(
            [
                self.missing_skills,
                self.recommended_technologies,
                self.learning_resources,
                self.practice_projects,
                self.improvement_path,
                self.sources,
                self.metadata,
            ]
        ):
            raise ValueError("not-started guidance cannot contain output")

        return self
