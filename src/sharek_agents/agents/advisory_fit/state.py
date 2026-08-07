from __future__ import annotations

from typing import TypedDict

from sharek_agents.agents.advisory_fit.schemas import (
    AdvisoryFitInput,
    AdvisoryFitResult,
    ApproachAnalysis,
    Assessment,
    EvidenceUnderstanding,
    LevelMatch,
    SkillMatch,
)


class SkillVerificationEntry(TypedDict, total=False):
    """Per-requirement contributor skill evidence, request-authoritative.

    ``contributor_level`` always comes from ``request.contributor.skills``;
    the LLM's ``contributor_level`` field is never trusted. Relevance is
    intentionally absent: it is determined by
    ``select_relevant_requirements`` and carried in its selection partitions
    (``relevant_requirements`` / ``partially_relevant_requirements``), never
    by the matching node.
    """

    contributor_level: str | None
    skill_match: SkillMatch
    explanation: str


class AgentState(TypedDict, total=False):
    """Bounded deterministic workflow state for the Advisory Fit agent.

    The original request remains immutable. ``approach_analysis`` is the
    structured understanding of the Contributor Approach produced by
    ``understand_approach``; ``intended_approach`` is the companion structured
    understanding of what the contributor intends to build (currently
    retained, not consumed). ``select_relevant_requirements`` writes the
    ordered selection partitions (``relevant_requirements`` /
    ``partially_relevant_requirements``) AND ``requirement_classifications`` —
    the explicit DIRECT / PARTIAL / NOT_MENTIONED decision for every project
    requirement, keyed by normalized requirement name. Selection is therefore
    the ONLY producer of the relevance classification;
    ``match_skills_and_evidence`` consumes the selection partitions and request
    data only; ``calculate_fit`` consumes ``requirement_classifications`` and
    the matching data and never re-classifies. All analytical fields are
    written from authoritative request data; the deterministic final result is
    assembled from them.
    """

    request: AdvisoryFitInput
    approach_analysis: EvidenceUnderstanding
    intended_approach: ApproachAnalysis
    relevant_requirements: list[str]
    partially_relevant_requirements: list[str]
    requirement_classifications: dict[str, str]
    skill_verification: dict[str, SkillVerificationEntry]
    level_evaluations: dict[str, LevelMatch]
    requirement_assessments: list[Assessment]
    fit_percentage: float
    summary: str
    final_result: AdvisoryFitResult