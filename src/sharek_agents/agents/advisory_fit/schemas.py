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

    @field_validator("assessment_request_id")
    @classmethod
    def request_id_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("assessment_request_id must not be blank")
        return value

    @field_validator("requirements")
    @classmethod
    def requirement_ids_must_be_unique(
        cls, value: list[RequirementSnapshot]
    ) -> list[RequirementSnapshot]:
        ids = [requirement.id for requirement in value]
        if len(ids) != len(set(ids)):
            raise ValueError("requirement snapshot IDs must be unique")
        return value

    @field_validator("allowed_evidence_ids")
    @classmethod
    def evidence_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("allowed evidence IDs must not be blank")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("allowed evidence IDs must be unique")
        return cleaned


class AdvisoryFitFinding(ContractModel):
    requirement_id: str = Field(min_length=1, max_length=200)
    requirement_kind: RequirementKind
    finding: FindingKind
    confidence: Confidence
    citations: list[str] = Field(min_length=1)
    uncertainty: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1, max_length=2000)

    @field_validator("requirement_id", "explanation")
    @classmethod
    def finding_text_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("citations")
    @classmethod
    def citations_must_be_unique_and_non_blank(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("citations must not be blank")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("citations must be unique")
        return cleaned


class AdvisoryFitProviderOutput(ContractModel):
    findings: list[AdvisoryFitFinding] = Field(min_length=1)


class AdvisoryFitMetadata(ContractModel):
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=100)
    prompt_version: str = Field(min_length=1, max_length=100)
    schema_version: str = Field(min_length=1, max_length=100)
    service_version: str = Field(min_length=1, max_length=100)
    latency_ms: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class AdvisoryFitResult(ContractModel):
    status: AssessmentStatus
    findings: list[AdvisoryFitFinding] = Field(default_factory=list)
    metadata: AdvisoryFitMetadata | None = None

    @model_validator(mode="after")
    def payload_matches_status(self) -> "AdvisoryFitResult":
        if self.status == "COMPLETED":
            if not self.findings:
                raise ValueError("completed Advisory Fit results require findings")
            if self.metadata is None:
                raise ValueError("completed Advisory Fit results require metadata")
        elif self.findings or self.metadata is not None:
            raise ValueError(
                "not-started Advisory Fit results cannot contain findings or metadata"
            )
        return self
