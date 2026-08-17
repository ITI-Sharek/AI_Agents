from __future__ import annotations

from pydantic import Field, field_validator

from sharek_agents.agents.advisory_fit.schemas import (
    AdvisoryFitResult,
    ContractModel,
)


class GapGuidanceInput(ContractModel):
    """Input contract for Gap Guidance.

    Phase 1 accepts the Advisory Fit result as-is: the endpoint works
    entirely from the supplied result and does not require project or
    contributor database identifiers.
    """

    advisory_fit_result: AdvisoryFitResult
    # Requested natural-language output language (free-form language name,
    # e.g. "arabic", "english", "spanish"; never an ISO code). Empty means
    # the default behavior (English natural-language output). The value only
    # controls the language of user-facing natural-language text
    # (learningGuidance and practiceRoadmap); it never affects the preserved
    # Advisory Fit values, identifiers, enum values, or levels.
    answer: str = Field(default="", max_length=100)

    @field_validator("answer")
    @classmethod
    def clean_answer(cls, value: str) -> str:
        return value.strip()


class GapGuidanceResult(ContractModel):
    """Result contract for Gap Guidance (Phase 4 combined response).

    The response preserves the exact ``AdvisoryFitResult`` the endpoint
    received (never recalculated) and adds ONE combined
    ``learning_guidance`` covering all relevant skill gaps and ONE combined
    ``practice_roadmap`` string ordering those gaps into a single coherent
    learning/practice path. Skills without a meaningful gap
    (MATCHED / EXACT) must not appear in the guidance or roadmap.
    """

    advisory_fit_result: AdvisoryFitResult
    learning_guidance: str = Field(min_length=1)
    practice_roadmap: str = Field(min_length=1)
