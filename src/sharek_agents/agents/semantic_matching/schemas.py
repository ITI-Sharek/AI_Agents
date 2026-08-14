"""Semantic Matching schemas (Phase 1 revision: pgvector-compatible storage).

Phase 1 models the matching-index records only: normal profile data plus
freshness and embedding metadata. The matching request contract
(``SemanticMatchRequest``) is the single matching endpoint body
(``POST /semantic-matching/match``); Phase 8 wires it to the production
matching flow and adds its response contracts (``ProjectMatch`` /
``SemanticMatchResponse``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


SkillLevel = Literal["beginner", "intermediate", "advanced"]


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
    """A skill together with its level (single structure, never split)."""

    skill: str = Field(min_length=1, max_length=200)
    level: SkillLevel

    @field_validator("skill")
    @classmethod
    def skill_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("skill must not be empty")
        return stripped


class Evidence(ContractModel):
    """A matching evidence record.

    Kept separate from skills because evidence may have its own structure;
    fields are optional except ``title`` so partial snapshots are accepted.
    """

    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(default="", max_length=10000)
    technologies: list[str] = Field(default_factory=list)
    description: str = Field(default="", max_length=10000)
    repository: str | None = Field(default=None, max_length=500)
    url: str | None = Field(default=None, max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _no_duplicate_skills(value: list[SkillItem]) -> list[SkillItem]:
    """Reject duplicate skills (case-insensitive) in any skill list."""
    seen: set[str] = set()
    for item in value:
        key = item.skill.casefold()
        if key in seen:
            raise ValueError(f"Duplicate skill: '{item.skill}'")
        seen.add(key)
    return value


class _MatchRecordBase(ContractModel):
    """Shared fields of a matching-index record.

    The original entity id is the primary identifier; a separate ``id``
    column is intentionally not added.

    Freshness: ``source_version`` / ``source_updated_at`` hold whatever
    authoritative version/update identifier the main database exposes, so
    the index can detect stale data. The main database is the source of
    truth; this feature never writes to it.

    Embedding metadata: ``embedding_model``, ``embedding_model_version``,
    and ``embedding_schema_version`` describe how a stored vector was
    generated so later phases can detect when vectors must be regenerated.
    All embedding fields are ``None`` until embedding generation exists.
    """

    skills: list[SkillItem] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    source_version: str | None = Field(default=None, max_length=200)
    source_updated_at: datetime | None = None

    embedding: list[float] | None = None
    embedding_model: str | None = Field(default=None, max_length=200)
    embedding_model_version: str | None = Field(default=None, max_length=200)
    embedding_schema_version: str | None = Field(default=None, max_length=200)

    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("skills")
    @classmethod
    def no_duplicate_skills(cls, value: list[SkillItem]) -> list[SkillItem]:
        return _no_duplicate_skills(value)


class _SourceDataBase(ContractModel):
    """Authoritative entity data read from the main database (source of truth).

    The Semantic Matching feature reads this data read-only through the
    source data provider; it never writes to the main database.

    ``source_version`` / ``source_updated_at`` are TODO placeholders: the
    main database is owned by the backend/NestJS side and its real field
    names are not available in this repository. These fields hold whatever
    authoritative version/update identifier the backend supplies later; they
    are deliberately NOT ``generation_id`` / ``requested_at`` (those are
    request-scoped and do not represent an approved/current profile).
    """

    skills: list[SkillItem] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    source_version: str | None = Field(default=None, max_length=200)
    source_updated_at: datetime | None = None

    @field_validator("skills")
    @classmethod
    def no_duplicate_skills(cls, value: list[SkillItem]) -> list[SkillItem]:
        return _no_duplicate_skills(value)


class ProjectSourceData(_SourceDataBase):
    """Authoritative Project data required for matching."""

    project_id: int


class ContributorSourceData(_SourceDataBase):
    """Authoritative Contributor data required for matching."""

    contributor_id: int


class ProjectMatchRecord(_MatchRecordBase):
    """A Project's matching-index record: profile data plus embedding metadata."""

    project_id: int


class ContributorMatchRecord(_MatchRecordBase):
    """A Contributor's matching-index record: profile data plus embedding metadata."""

    contributor_id: int


class SemanticMatchRequest(ContractModel):
    """Matching request body for ``POST /semantic-matching/match`` (Phase 8).

    Accepts ONLY the query entity id and ``top_k``. Exactly one of
    ``contributor_id`` / ``project_id`` is required; the direction is
    determined by which id is present (contributor_id -> matching Projects,
    project_id -> matching Contributors). Skills, levels, and evidence are
    never accepted from the client; the matching service reads them from
    its own matching index.

    Phase 8 implements the Contributor -> Projects direction only.
    """

    contributor_id: int | None = Field(
        default=None,
        description="Query Contributor id (Contributor -> Projects direction)",
    )
    project_id: int | None = Field(
        default=None,
        description="Query Project id (Project -> Contributors direction)",
    )
    top_k: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def exactly_one_entity_id(self) -> SemanticMatchRequest:
        is_contributor = self.contributor_id is not None
        is_project = self.project_id is not None
        if is_contributor == is_project:
            raise ValueError(
                "Exactly one of contributor_id or project_id must be provided"
            )
        return self


class ProjectMatch(ContractModel):
    """One ranked Project match for a Contributor query."""

    project_id: int
    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    rank: int = Field(ge=1)


class SemanticMatchResponse(ContractModel):
    """Ranked matching Projects for a Contributor (Phase 8).

    ``matches`` holds at most ``top_k`` Projects, ranked by cosine
    similarity descending (``rank`` starts at 1). Fewer Projects may be
    returned when the matching index holds fewer indexed Projects.
    """

    contributor_id: int
    top_k: int
    matches: list[ProjectMatch] = Field(default_factory=list)