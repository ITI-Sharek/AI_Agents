from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


RequirementKind = Literal["required", "preferred"]

# The three platform proficiency levels, and only these. The backend compares a
# contributor's approved proficiency against this value using a total order, and
# a fourth level would have no defined position in it.
RequiredLevel = Literal["beginner", "intermediate", "advanced"]

# Categorical, never a percentage. A number invites the reader to treat an
# inferred level as a measurement; it is not one, and DEC-010 forbids presenting
# fit as a number anywhere in the product.
Confidence = Literal["high", "medium", "low"]

Difficulty = Literal["beginner", "intermediate", "advanced"]

# The same cap the backend enforces. A Request claiming to need more than
# fifteen distinct skills at stated levels is describing a project rather than a
# task, and the resulting block would be unexplainable to the contributor.
MAX_INFERRED_SKILLS = 15


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class ContractModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        # THE PRIVACY BOUNDARY. `forbid` is what makes "this endpoint never sees
        # contributor data" a property of the contract rather than a promise in
        # a document: a caller that adds `contributorId` to the payload gets a
        # 422, so the field cannot arrive here by accident or by a future
        # backend change nobody reviewed.
        extra="forbid",
    )


class RequirementInferenceInput(ContractModel):
    """The bounded Contribution Request content, and nothing else.

    There is deliberately no contributor identifier, no approved-skill list, and
    no Application reference anywhere in this schema. The agent is asked what
    the *work* demands; who might do it is not its business and cannot be made
    its business without changing this class.
    """

    contribution_request_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=255)
    description: str = Field(min_length=1, max_length=5000)
    requirement_texts: list[str] = Field(default_factory=list, max_length=40)
    technology_tags: list[str] = Field(default_factory=list, max_length=20)
    difficulty: Difficulty | None = None
    contract_version: Literal["requirement-inference-v1"]

    @field_validator("contribution_request_id", "title", "description")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("requirement_texts", "technology_tags")
    @classmethod
    def entries_not_blank(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("entries must not be blank")
        if any(len(item) > 500 for item in cleaned):
            raise ValueError("entries must be at most 500 characters")
        return cleaned


class InferredSkillRequirement(ContractModel):
    skill_name: str = Field(min_length=1, max_length=100)
    required_level: RequiredLevel
    kind: RequirementKind
    confidence: Confidence
    rationale: str = Field(min_length=1, max_length=1000)

    @field_validator("skill_name")
    @classmethod
    def normalized_skill_name(cls, value: str) -> str:
        """Lowercase and trimmed, so the caller receives one spelling per skill.

        Applied in the schema rather than in the service so it holds for
        provider output and for anything constructed in a test — there is no
        path that produces an unnormalized name.
        """
        value = " ".join(value.split()).casefold()
        if not value:
            raise ValueError("skill name must not be blank")
        return value

    @field_validator("rationale")
    @classmethod
    def rationale_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("rationale must not be blank")
        return value


class RequirementInferenceProviderOutput(ContractModel):
    """What the model is asked for, before the service enforces the cap.

    Unbounded here on purpose: a model that returns forty skills has produced
    something the service must *truncate deterministically*, not something the
    schema should reject outright — rejecting would turn a verbose answer into a
    502 and leave the owner with nothing.
    """

    skills: list[InferredSkillRequirement] = Field(default_factory=list)


class RequirementInferenceMetadata(ContractModel):
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=100)
    prompt_version: Literal["requirement-inference-v1"]
    schema_version: Literal["requirement-inference-v1"]
    service_version: str = Field(min_length=1, max_length=100)
    latency_ms: int = Field(ge=0)


class RequirementInferenceResult(ContractModel):
    """Findings about the work. Never a verdict about a person.

    There is no `eligible`, `blocked`, `score`, or `rank` field, and no place to
    put one: the split ADR 0001 set for Advisory Fit holds here too, and this is
    the schema that enforces it.
    """

    skills: list[InferredSkillRequirement] = Field(
        default_factory=list, max_length=MAX_INFERRED_SKILLS
    )
    metadata: RequirementInferenceMetadata
