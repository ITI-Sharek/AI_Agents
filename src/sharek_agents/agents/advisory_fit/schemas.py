from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SkillLevel = Literal["beginner", "intermediate", "advanced"]
SkillMatch = Literal["MATCHED", "NOT_MATCHED", "NOT_EVIDENCED"]
ApproachRelevance = Literal["DIRECT", "PARTIAL", "NOT_MENTIONED", "UNCLEAR"]
LevelMatch = Literal["EXACT", "HIGHER", "LOWER", "MISSING"]


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class ContractModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class SkillItem(ContractModel):
    skill: str = Field(min_length=1, max_length=200)
    level: SkillLevel

    @field_validator("skill")
    @classmethod
    def skill_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("skill must not be empty")
        return stripped


class AdvisoryFitInput(ContractModel):
    project_requirements: list[SkillItem] = Field(min_length=1)
    contributor_skills: list[SkillItem] = Field(default_factory=list)
    contributor_approach: str = Field(default="", max_length=10000)

    @field_validator("project_requirements")
    @classmethod
    def no_duplicate_requirements(cls, value: list[SkillItem]) -> list[SkillItem]:
        seen: set[str] = set()
        for item in value:
            key = item.skill.casefold()
            if key in seen:
                raise ValueError(
                    f"Duplicate project requirement skill: '{item.skill}'"
                )
            seen.add(key)
        return value

    @field_validator("contributor_skills")
    @classmethod
    def no_conflicting_contributor_skills(
        cls, value: list[SkillItem]
    ) -> list[SkillItem]:
        seen: dict[str, SkillItem] = {}
        for item in value:
            key = item.skill.casefold()
            existing = seen.get(key)
            if existing is not None and existing.level != item.level:
                raise ValueError(
                    f"Conflicting levels for skill '{item.skill}': "
                    f"'{existing.level}' and '{item.level}'"
                )
            seen.setdefault(key, item)
        return value


class RequirementAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str
    required_level: str
    contributor_level: str | None = None
    skill_match: SkillMatch
    approach_relevance: ApproachRelevance
    explanation: str


class AdvisoryFitAIOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessments: list[RequirementAnalysis]

    @field_validator("assessments")
    @classmethod
    def no_duplicate_skills(cls, value: list[RequirementAnalysis]) -> list[RequirementAnalysis]:
        seen: set[str] = set()
        for item in value:
            key = item.skill.casefold()
            if key in seen:
                raise ValueError(
                    f"Duplicate skill in AI analysis: '{item.skill}'"
                )
            seen.add(key)
        return value


class Assessment(ContractModel):
    skill: str
    required_level: str
    contributor_level: str | None = None
    skill_match: SkillMatch
    level_match: LevelMatch
    approach_relevance: ApproachRelevance
    explanation: str


class AdvisoryFitResult(ContractModel):
    fit_percentage: float = Field(ge=0.0, le=100.0)
    assessments: list[Assessment]
    summary: str
