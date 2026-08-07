from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SkillLevel = Literal["beginner", "intermediate", "advanced"]
SkillMatch = Literal["MATCHED", "NOT_MATCHED", "NOT_EVIDENCED"]
ApproachRelevance = Literal["DIRECT", "PARTIAL", "NOT_MENTIONED", "UNCLEAR"]
LevelMatch = Literal["EXACT", "HIGHER", "LOWER", "MISSING"]
ConfidenceLevel = Literal["high", "medium", "low"]


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


class ProjectInfo(ContractModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=10000)


class Evidence(ContractModel):
    """Structured contributor evidence snapshot, normalized for AI reasoning.

    ``title`` is the only required field; every other field is optional so the
    backend may supply partial snapshots. ``metadata`` carries purpose-specific
    extra data without loosening the declared contract (``extra="forbid"``).
    """

    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(default="", max_length=10000)
    technologies: list[str] = Field(default_factory=list)
    description: str = Field(default="", max_length=10000)
    repository: str | None = Field(default=None, max_length=500)
    url: str | None = Field(default=None, max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContributorInfo(ContractModel):
    skills: list[SkillItem] = Field(default_factory=list)
    approach: str = Field(default="", max_length=10000)
    evidence: list[Evidence] = Field(default_factory=list)

    @field_validator("skills")
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


class AdvisoryFitInput(ContractModel):
    project: ProjectInfo
    project_requirements: list[SkillItem] = Field(min_length=1)
    contributor: ContributorInfo

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


class EvidenceReferenced(ContractModel):
    """Base for extracted evidence-understanding items that cite evidence.

    Every extracted item carries ``evidence_indexes``: the 0-based indexes of
    the ``contributor.evidence`` records that support it. The schema enforces
    non-negative, unique indexes; the upper bound is checked deterministically
    by the node against the actual evidence list (fail closed).
    """

    evidence_indexes: list[int] = Field(default_factory=list)

    @field_validator("evidence_indexes")
    @classmethod
    def valid_evidence_indexes(
        cls, value: list[int]
    ) -> list[int]:
        if any(index < 0 for index in value):
            raise ValueError("evidence indexes must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("evidence indexes must be unique")
        return value


class BuiltArtifact(EvidenceReferenced):
    """What the contributor has built, as evidenced by one or more records."""

    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(default="", max_length=1000)
    technologies: list[str] = Field(default_factory=list)

    @field_validator("technologies")
    @classmethod
    def clean_artifact_technologies(
        cls, value: list[str]
    ) -> list[str]:
        cleaned = [t.strip() for t in value]
        if any(not t for t in cleaned):
            raise ValueError("artifact technologies must not be empty")
        seen: set[str] = set()
        for t in cleaned:
            key = t.casefold()
            if key in seen:
                raise ValueError(
                    f"duplicate artifact technology: '{t}'"
                )
            seen.add(key)
        return cleaned


class DemonstratedCapability(EvidenceReferenced):
    """A technical capability the evidence actually demonstrates."""

    capability: str = Field(min_length=1, max_length=200)
    confidence: ConfidenceLevel


class ArchitecturalPattern(EvidenceReferenced):
    """An architectural or design pattern that appears in the evidence."""

    pattern: str = Field(min_length=1, max_length=200)


class EvidencedTechnology(EvidenceReferenced):
    """A technology that is actually evidenced by the records."""

    name: str = Field(min_length=1, max_length=200)


class SupportedExperience(EvidenceReferenced):
    """An experience statement directly supported by the evidence."""

    experience: str = Field(min_length=1, max_length=500)


class EvidenceUnderstanding(ContractModel):
    """Structured intermediate representation of the Contributor Evidence.

    Produced by the ``understand_approach`` workflow node and independent of
    project requirements: it records what the contributor has built, the
    technical capabilities the evidence demonstrates (each with a confidence),
    the architectural patterns that appear, the technologies that are actually
    evidenced, and the experience that is directly supported. Later nodes
    consume this representation.
    """

    summary: str = Field(default="", max_length=2000)
    built_artifacts: list[BuiltArtifact] = Field(default_factory=list)
    capabilities: list[DemonstratedCapability] = Field(default_factory=list)
    architectural_patterns: list[ArchitecturalPattern] = Field(
        default_factory=list
    )
    technologies: list[EvidencedTechnology] = Field(default_factory=list)
    supported_experience: list[SupportedExperience] = Field(
        default_factory=list
    )


class ApproachAnalysis(ContractModel):
    """Structured understanding of what the contributor intends to build.

    Produced by the ``understand_approach`` workflow node from the
    Contributor Approach (``contributor.approach``) as the primary subject,
    with the Contributor Evidence as supporting context only. It records the
    intended work — features, capabilities, architecture, technologies, and
    implementation plan, each with a confidence — and never project
    requirements, selected requirements, contributor skills, fit scores,
    recommendations, or roadmaps. Later nodes may consume this
    representation; the current workflow does not depend on it.
    """

    summary: str = Field(default="", max_length=2000)
    intended_features: list[str] = Field(default_factory=list)
    intended_capabilities: list[str] = Field(default_factory=list)
    intended_architecture: list[str] = Field(default_factory=list)
    intended_technologies: list[str] = Field(default_factory=list)
    implementation_plan: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = "low"

    @field_validator(
        "intended_features",
        "intended_capabilities",
        "intended_architecture",
        "intended_technologies",
        "implementation_plan",
    )
    @classmethod
    def clean_approach_items(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("approach items must not be empty")
        if any(len(item) > 200 for item in cleaned):
            raise ValueError("approach items must be at most 200 characters")
        seen: set[str] = set()
        for item in cleaned:
            key = item.casefold()
            if key in seen:
                raise ValueError(f"duplicate approach item: '{item}'")
            seen.add(key)
        return cleaned
