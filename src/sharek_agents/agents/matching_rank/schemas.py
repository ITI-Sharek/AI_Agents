from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Proficiency = Literal["beginner", "intermediate", "advanced"]

# The categorical band the backend already computed. It is echoed back untouched
# so the agent can read it while reasoning; there is no field for the agent to
# state a band of its own, and no numeric field anywhere in this contract.
Confidence = Literal["HIGH", "MEDIUM", "LOW"]

# The most candidates one shortlist may carry. The backend caps a Gold
# contributor at ten matched projects, so this is already far above what it will
# send; the bound exists so a malformed or hostile caller cannot make this
# endpoint do unbounded model work.
MAX_CANDIDATES = 50

# A narrative is one or two sentences a contributor reads on a card. Longer than
# this is not a better explanation, it is an essay in a 300px box.
MAX_NARRATIVE_LENGTH = 300


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class ContractModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        # `forbid` is what keeps this endpoint incapable of receiving or
        # returning a score. A caller adding `matchScore`, or a model persuaded
        # to emit one, gets a 422 rather than a field that quietly reaches a
        # contributor's screen (DEC-010).
        extra="forbid",
    )


class MatchedSkill(ContractModel):
    """One approved skill the backend already matched to a candidate."""

    name: str = Field(min_length=1, max_length=100)
    proficiency: Proficiency

    @field_validator("name")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("skill name must not be blank")
        return value


class MatchCandidate(ContractModel):
    """One shortlisted Contribution Request, with the facts the backend derived.

    Everything here is a fact the backend computed deterministically and is
    already willing to show this contributor. There is no contributor
    identifier, no evidence blob, and no field describing anyone's eligibility:
    the agent is ordering work, not judging a person.
    """

    request_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=255)
    project_name: str = Field(default="", max_length=255)
    technology_tags: list[str] = Field(default_factory=list, max_length=20)
    requirement_texts: list[str] = Field(default_factory=list, max_length=40)
    matched_skills: list[MatchedSkill] = Field(default_factory=list, max_length=50)
    # The backend's own band and position. Echoed so the agent can weigh them;
    # it may disagree by reordering, which is the entire point of the endpoint.
    confidence: Confidence
    deterministic_rank: int = Field(ge=1)


class MatchingRankInput(ContractModel):
    """A shortlist the backend already computed, authorized and capped.

    The agent never discovers candidates. It cannot: there is no query, no
    database handle and no contributor id in this schema, only the finished list
    and the skills that produced it.
    """

    matching_request_id: str = Field(min_length=1, max_length=200)
    approved_skills: list[MatchedSkill] = Field(default_factory=list, max_length=50)
    candidates: list[MatchCandidate] = Field(min_length=1, max_length=MAX_CANDIDATES)
    contract_version: Literal["matching-rank-v1"]

    @model_validator(mode="after")
    def request_ids_are_unique(self) -> MatchingRankInput:
        ids = [candidate.request_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidates must not repeat a requestId")
        return self


class RankedMatch(ContractModel):
    """One item of the returned order.

    Deliberately only two fields. Anything the agent could add here — a score, a
    band of its own, a verdict — is a thing the backend would have to decide
    whether to trust. It cannot be added by accident because the model is
    `extra="forbid"`.
    """

    request_id: str = Field(min_length=1, max_length=200)
    why_this_matches: str = Field(min_length=1, max_length=MAX_NARRATIVE_LENGTH)

    @field_validator("why_this_matches")
    @classmethod
    def narrative_is_clean(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("whyThisMatches must not be blank")
        # A percentage is forbidden presentation (DEC-010), and it is easier to
        # refuse it here than to hope no prompt ever elicits one.
        if "%" in value:
            raise ValueError("whyThisMatches must not contain a percentage")
        return value


class MatchingRankProviderOutput(ContractModel):
    """What the model is asked for, before the service checks the id set."""

    matches: list[RankedMatch] = Field(default_factory=list)


class MatchingRankMetadata(ContractModel):
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=100)
    prompt_version: Literal["matching-rank-v1"]
    schema_version: Literal["matching-rank-v1"]
    service_version: str = Field(min_length=1, max_length=100)
    latency_ms: int = Field(ge=0)


class MatchingRankResult(ContractModel):
    """An order and an explanation. Never a score, never a verdict.

    `matches` is a permutation of the input `candidates` — the service enforces
    that before returning, and the backend enforces it again on receipt. There
    is no `rank` field: position in this list *is* the rank, so the two cannot
    disagree with each other.
    """

    matches: list[RankedMatch]
    metadata: MatchingRankMetadata
