from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


RequirementKind = Literal["required", "preferred"]
FindingKind = Literal[
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "NOT_EVIDENCED",
    "INCONCLUSIVE",
]
Confidence = Literal["HIGH", "MEDIUM", "LOW"]
AssessmentStatus = Literal[
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


class RequirementSnapshot(ContractModel):
    id: str = Field(min_length=1, max_length=200)
    kind: RequirementKind
    position: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=5000)

    @field_validator("id", "text")
    @classmethod
    def required_text_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class AdvisoryFitInput(ContractModel):
    assessment_request_id: str = Field(min_length=1, max_length=200)
    requirements: list[RequirementSnapshot] = Field(min_length=1)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    allowed_evidence_ids: list[str] = Field(default_factory=list)
    requested_at: datetime
    contract_version: Literal["advisory-fit-v1"]

    @field_validator("requirements")
    @classmethod
    def unique_requirements(
        cls, value: list[RequirementSnapshot]
    ) -> list[RequirementSnapshot]:
        identifiers = [item.id for item in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("requirement snapshot IDs must be unique")
        return value

    @field_validator("allowed_evidence_ids")
    @classmethod
    def unique_evidence(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("allowed evidence IDs must be unique and non-blank")
        return cleaned


class AdvisoryFitFinding(ContractModel):
    requirement_id: str = Field(min_length=1, max_length=200)
    requirement_kind: RequirementKind
    finding: FindingKind
    confidence: Confidence
    citations: list[str] = Field(min_length=1)
    uncertainty: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1, max_length=2000)

    @field_validator("citations")
    @classmethod
    def unique_citations(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("citations must be unique and non-blank")
        return cleaned


class AdvisoryFitProviderOutput(ContractModel):
    findings: list[AdvisoryFitFinding] = Field(min_length=1)


class AdvisoryFitMetadata(ContractModel):
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=100)
    prompt_version: Literal["advisory-fit-v1"]
    schema_version: Literal["advisory-fit-v1"]
    service_version: str = Field(min_length=1, max_length=100)
    latency_ms: int = Field(ge=0)


class AdvisoryFitResult(ContractModel):
    status: AssessmentStatus
    findings: list[AdvisoryFitFinding] = Field(default_factory=list)
    metadata: AdvisoryFitMetadata | None = None

    @model_validator(mode="after")
    def payload_matches_status(self) -> "AdvisoryFitResult":
        if self.status == "COMPLETED":
            if not self.findings or self.metadata is None:
                raise ValueError("completed results require findings and metadata")
        elif self.findings or self.metadata is not None:
            raise ValueError("not-started results cannot contain analysis")
        return self
