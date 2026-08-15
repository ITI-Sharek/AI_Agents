from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from sharek_agents.agents.skill_gap_guidance.schemas import ContractModel


RequirementKind = Literal["required", "preferred"]
Proficiency = Literal["beginner", "intermediate", "advanced"]
EvidenceType = Literal[
    "approved_skill",
    "contribution_requirement",
    "reputation_signal",
    "retrieved_evidence",
]
Confidence = Literal["HIGH", "MEDIUM", "LOW"]
MatchingStatus = Literal[
    "COMPLETED",
    "NOT_STARTED_SYSTEM_LIMIT",
    "NOT_STARTED_NO_CANDIDATES",
]


class MatchingRequirementSnapshot(ContractModel):
    id: str = Field(min_length=1, max_length=200)
    kind: RequirementKind
    position: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=500)


class MatchingApprovedSkillSnapshot(ContractModel):
    skill_profile_id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=200)
    proficiency: Proficiency
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    evidence_summary: str | None = Field(default=None, max_length=2000)

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_are_unique(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("skill evidence IDs must be unique and non-blank")
        return cleaned


class MatchingReputationSnapshot(ContractModel):
    rating: float | None = Field(default=None, ge=0, le=5)
    completed_contributions: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=100)
    top_verified_skills: list[str] = Field(default_factory=list, max_length=20)


class MatchingCandidateSnapshot(ContractModel):
    contributor_id: str = Field(min_length=1, max_length=200)
    display_name: str = Field(min_length=1, max_length=200)
    username: str | None = Field(default=None, max_length=100)
    approved_skills: list[MatchingApprovedSkillSnapshot] = Field(
        min_length=1, max_length=100
    )
    reputation: MatchingReputationSnapshot


class MatchingEvidenceCapsule(ContractModel):
    evidence_id: str = Field(min_length=1, max_length=200)
    type: EvidenceType
    label: str = Field(min_length=1, max_length=300)
    summary: str | None = Field(default=None, max_length=2000)
    contributor_id: str | None = Field(default=None, max_length=200)


class ContributorMatchingInput(ContractModel):
    matching_request_id: str = Field(min_length=1, max_length=200)
    contribution_request_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=255)
    description: str = Field(min_length=1, max_length=5000)
    requirements: list[MatchingRequirementSnapshot] = Field(min_length=1, max_length=100)
    candidates: list[MatchingCandidateSnapshot] = Field(max_length=500)
    evidence: list[MatchingEvidenceCapsule] = Field(max_length=1000)
    allowed_evidence_ids: list[str] = Field(max_length=1000)
    requested_at: datetime
    contract_version: Literal["contributor-matching-v1"]

    @field_validator("requirements")
    @classmethod
    def requirement_ids_are_unique(
        cls, value: list[MatchingRequirementSnapshot]
    ) -> list[MatchingRequirementSnapshot]:
        ids = [item.id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("matching requirement IDs must be unique")
        return value

    @field_validator("candidates")
    @classmethod
    def candidate_ids_are_unique(
        cls, value: list[MatchingCandidateSnapshot]
    ) -> list[MatchingCandidateSnapshot]:
        ids = [item.contributor_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("matching candidate IDs must be unique")
        return value

    @field_validator("allowed_evidence_ids")
    @classmethod
    def allowed_evidence_ids_are_unique(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("allowed evidence IDs must be unique and non-blank")
        return cleaned

    @model_validator(mode="after")
    def evidence_scope_is_closed(self) -> "ContributorMatchingInput":
        allowed = set(self.allowed_evidence_ids)
        supplied = {item.evidence_id for item in self.evidence}
        if supplied != allowed:
            raise ValueError("evidence and allowed evidence IDs must have the same scope")
        skill_ids = {
            evidence_id
            for candidate in self.candidates
            for skill in candidate.approved_skills
            for evidence_id in skill.evidence_ids
        }
        if not skill_ids.issubset(allowed):
            raise ValueError("candidate skill evidence must be in the allowed scope")
        return self

class ContributorMatchingMatchedSkill(ContractModel):
    name: str = Field(min_length=1, max_length=200)
    proficiency: Proficiency
    evidence_ids: list[str] = Field(min_length=1, max_length=50)

    @field_validator("evidence_ids")
    @classmethod
    def matched_evidence_ids_are_unique(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("matched evidence IDs must be unique and non-blank")
        return cleaned


class ContributorMatchingProviderMatch(ContractModel):
    contributor_id: str = Field(min_length=1, max_length=200)
    match_score: float = Field(ge=0, le=1)
    confidence: Confidence
    justification: str = Field(min_length=1, max_length=2000)
    matched_skills: list[ContributorMatchingMatchedSkill] = Field(
        min_length=1, max_length=50
    )
    evidence_ids: list[str] = Field(min_length=1, max_length=100)

    @field_validator("evidence_ids")
    @classmethod
    def match_evidence_ids_are_unique(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("match evidence IDs must be unique and non-blank")
        return cleaned


class ContributorMatchingProviderOutput(ContractModel):
    matches: list[ContributorMatchingProviderMatch] = Field(max_length=500)


class ContributorMatchingMetadata(ContractModel):
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=100)
    prompt_version: str = Field(min_length=1, max_length=100)
    schema_version: str = Field(min_length=1, max_length=100)
    service_version: str = Field(min_length=1, max_length=100)
    latency_ms: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class ContributorMatchingResult(ContractModel):
    status: MatchingStatus
    matches: list[ContributorMatchingProviderMatch] = Field(default_factory=list)
    metadata: ContributorMatchingMetadata | None = None

    @model_validator(mode="after")
    def payload_matches_status(self) -> "ContributorMatchingResult":
        if self.status == "COMPLETED":
            if self.metadata is None:
                raise ValueError("completed matching results require metadata")
        elif self.matches or self.metadata is not None:
            raise ValueError("not-started matching results cannot contain matches or metadata")
        return self
