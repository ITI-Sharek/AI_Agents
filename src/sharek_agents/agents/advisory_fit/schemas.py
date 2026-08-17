from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The authoritative level hierarchy is beginner < intermediate < advanced <
# expert; rank order is owned by ``scoring._LEVEL_ORDER`` and level matching
# is deterministic Python (``scoring.calculate_level_match``) — never the LLM.
SkillLevel = Literal["beginner", "intermediate", "advanced", "expert"]
# Semantic skill-match state, classified by the LLM and validated by Python:
# ``MATCHED`` — the contributor explicitly has the required skill itself;
# ``RELATED`` — the contributor has one or more meaningfully related/adjacent
# declared skills (a related technology, framework, library, language, tool,
# methodology, or adjacent capability), never reported as a direct match;
# ``MISSING`` — no declared contributor skill is meaningfully related. Python
# validates and defaults, but never overrides the LLM's classification.
SkillMatch = Literal["MATCHED", "RELATED", "MISSING"]
# Semantic evidence-match state, classified by the LLM and validated by
# Python: ``MATCHED`` — the evidence directly proves usage/demonstration of
# the required skill itself; ``RELATED`` — the evidence demonstrates a
# related capability without directly proving the exact required skill;
# ``MISSING`` — the evidence provides no meaningful support. The LLM never
# emits scores or level information.
EvidenceSupport = Literal["MATCHED", "RELATED", "MISSING"]
ApproachRelevance = Literal["DIRECT", "PARTIAL", "NOT_MENTIONED", "UNCLEAR"]
# Semantic approach-to-requirement relevance, classified by the LLM and
# mapped to the response ``ApproachRelevance`` values by Python:
# ``DIRECT`` — the Approach directly requires, explicitly targets, or clearly
# describes work covered by the requirement; ``RELATED`` — the Approach does
# not directly name or target the exact requirement, but the described work is
# meaningfully related to it; ``NOT_RELEVANT`` — the described work has no
# meaningful relationship to the requirement. Python validates references and
# maps these verdicts to the existing response representation.
ApproachRequirementRelevance = Literal["DIRECT", "RELATED", "NOT_RELEVANT"]
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
    # Requested natural-language output language (free-form language name,
    # e.g. "arabic", "english", "spanish"; never an ISO code). Empty means
    # the default behavior (English natural-language output). The value only
    # controls the language of user-facing natural-language text; it never
    # affects identifiers, enum values, levels, scoring, or business logic.
    answer: str = Field(default="", max_length=100)

    @field_validator("answer")
    @classmethod
    def clean_answer(cls, value: str) -> str:
        return value.strip()

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
    evidence_match: EvidenceSupport = "MISSING"
    approach_relevance: ApproachRelevance
    explanation: str


class AdvisoryFitResult(ContractModel):
    fit_percentage: float = Field(ge=0.0, le=100.0)
    # Descriptive response metadata, computed deterministically by Python from
    # the per-requirement assessments: ``evaluated_skills`` is the number of
    # requirements actually evaluated (selected as relevant to the described
    # work — DIRECT or RELATED approach relevance, never NOT_MENTIONED), and
    # ``matched_skills`` is the number of evaluated requirements whose
    # ``skill_match`` is ``MATCHED`` (never RELATED or MISSING). These are
    # metadata only — they never feed the fit percentage or any score.
    matched_skills: int = Field(default=0, ge=0)
    evaluated_skills: int = Field(default=0, ge=0)
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


class ApproachRequirementRelation(ContractModel):
    """LLM-classified relevance of one authoritative project requirement.

    Produced by the approach-analysis LLM call in ``understand_approach``.
    ``requirement`` is the exact authoritative project requirement skill
    identifier from the request; ``relevance`` is ``DIRECT`` / ``RELATED`` /
    ``NOT_RELEVANT``. The LLM classifies the relationship only — it never
    supplies levels, scores, or authoritative identifiers, and it must never
    invent requirements. Python validates every reference against the
    authoritative project requirements and fails closed.
    """

    requirement: str = Field(min_length=1, max_length=200)
    relevance: ApproachRequirementRelevance
    explanation: str = Field(default="", max_length=500)

    @field_validator("requirement")
    @classmethod
    def requirement_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("requirement must not be empty")
        return stripped


class ApproachAnalysis(ContractModel):
    """Structured understanding of the work described by the Approach.

    Produced by the ``understand_approach`` workflow node from the Approach
    (``contributor.approach``) as the primary subject, with the Contributor
    Evidence as supporting context only. The Approach is a free-text
    description of the work / requested contribution / technical intent; it
    may be written by the Project Owner, the Contributor, or another
    authorized source, so it must never be assumed to be the contributor's
    own technical plan. This records the described work — features,
    capabilities, architecture, technologies, and implementation steps, each
    with a confidence — and, in ``requirement_relations``, the semantic
    relevance (``DIRECT`` / ``RELATED`` / ``NOT_RELEVANT``) of each
    authoritative project requirement to the described work. It never
    contains selected requirements, contributor skills, fit scores,
    recommendations, or roadmaps. Later nodes may consume this
    representation.
    """

    summary: str = Field(default="", max_length=2000)
    intended_features: list[str] = Field(default_factory=list)
    intended_capabilities: list[str] = Field(default_factory=list)
    intended_architecture: list[str] = Field(default_factory=list)
    intended_technologies: list[str] = Field(default_factory=list)
    implementation_plan: list[str] = Field(default_factory=list)
    requirement_relations: list[ApproachRequirementRelation] = Field(
        default_factory=list
    )
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


class RequirementSkillRelation(ContractModel):
    """LLM-classified semantic relation for one requirement skill.

    Produced by the relation-classification LLM call in
    ``match_skills_and_evidence``. ``requirement_skill`` is the exact
    authoritative requirement skill identifier from the request;
    ``relation`` is ``MATCHED`` / ``RELATED`` / ``MISSING``;
    ``related_skills`` lists the exact declared contributor skill names that
    support a ``RELATED`` classification. The LLM classifies the relationship
    only — it never supplies levels, scores, or authoritative identifiers.
    """

    requirement_skill: str = Field(min_length=1, max_length=200)
    relation: SkillMatch
    related_skills: list[str] = Field(default_factory=list)
    explanation: str = Field(default="", max_length=500)

    @field_validator("requirement_skill")
    @classmethod
    def requirement_skill_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("requirement_skill must not be empty")
        return stripped

    @field_validator("related_skills")
    @classmethod
    def clean_related_skills(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("related_skills must not be empty")
        seen: set[str] = set()
        for item in cleaned:
            key = item.casefold()
            if key in seen:
                raise ValueError(f"duplicate related skill: '{item}'")
            seen.add(key)
        return cleaned


class RequirementEvidenceRelation(EvidenceReferenced):
    """LLM-classified semantic evidence relation for one requirement skill.

    Produced by the relation-classification LLM call in
    ``match_skills_and_evidence``. ``requirement_skill`` is the exact
    authoritative requirement skill identifier; ``relation`` is ``MATCHED`` /
    ``RELATED`` / ``MISSING``; ``evidence_indexes`` are 0-based indexes of
    the evidence records that support the relation (range-checked
    deterministically by the node). The LLM classifies the relationship only.
    """

    requirement_skill: str = Field(min_length=1, max_length=200)
    relation: EvidenceSupport
    explanation: str = Field(default="", max_length=500)

    @field_validator("requirement_skill")
    @classmethod
    def requirement_skill_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("requirement_skill must not be empty")
        return stripped


class RequirementRelationAnalysis(ContractModel):
    """Bounded structured output of the relation-classification LLM call.

    One record per relevant requirement per section; ``skill_relations``
    classifies requirement ↔ declared-skill relationships and
    ``evidence_relations`` classifies requirement ↔ evidence relationships.
    Records are strictly validated by the workflow node (requirement
    references, declared-skill references, evidence indexes, duplicates,
    caps); requirements not covered by a record default to ``MISSING``.
    """

    skill_relations: list[RequirementSkillRelation] = Field(
        default_factory=list
    )
    evidence_relations: list[RequirementEvidenceRelation] = Field(
        default_factory=list
    )
