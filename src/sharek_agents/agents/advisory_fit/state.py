from __future__ import annotations

from typing import TypedDict

from sharek_agents.agents.advisory_fit.schemas import (
    AdvisoryFitInput,
    AdvisoryFitResult,
    ApproachAnalysis,
    Assessment,
    EvidenceSupport,
    EvidenceUnderstanding,
    LevelMatch,
    SkillMatch,
)


class SkillVerificationEntry(TypedDict, total=False):
    """Per-requirement contributor skill evidence, request-authoritative.

    ``contributor_level`` always comes from ``request.contributor.skills``
    (the entry whose normalized name equals the requirement name); the LLM's
    ``contributor_level`` field is never trusted. ``skill_match`` is the
    LLM-classified semantic skill relation (``MATCHED`` / ``RELATED`` /
    ``MISSING``), validated by Python and defaulted to ``MISSING`` when the
    relation analysis does not cover the requirement. ``evidence_match`` is
    the LLM-classified semantic evidence relation (``MATCHED`` / ``RELATED``
    / ``MISSING``) over the structured ``EvidenceUnderstanding``
    (``approach_analysis``); it never replaces or rewrites the declared skill
    or level. The three dimensions stay independent: a skill can be MATCHED
    with MISSING evidence, RELATED with MATCHED support, and so on. A
    ``RELATED`` classification never creates or implies a level match — the
    level always comes from the authoritative skill list entry whose name
    equals the requirement name. Relevance is intentionally absent: it is
    determined by ``select_relevant_requirements`` and carried in its
    selection partitions (``relevant_requirements`` /
    ``partially_relevant_requirements``), never by the matching node.
    """

    contributor_level: str | None
    skill_match: SkillMatch
    evidence_match: EvidenceSupport
    explanation: str


class AgentState(TypedDict, total=False):
    """Bounded deterministic workflow state for the Advisory Fit agent.

    The original request remains immutable. ``approach_analysis`` is the
    structured evidence understanding (``EvidenceUnderstanding``) produced by
    ``understand_approach`` and consumed as input to the relation
    classification in ``match_skills_and_evidence`` (reused, never duplicated
    or re-extracted); ``intended_approach`` is the companion structured
    understanding of the work described by the Approach — which may be
    written by the Project Owner, the Contributor, or another authorized
    source — produced by ``understand_approach`` and carrying the
    LLM-classified semantic relevance of each authoritative project
    requirement (``ApproachAnalysis.requirement_relations``). It is consumed
    by ``select_relevant_requirements``, which maps those LLM verdicts to the
    response representation. ``select_relevant_requirements`` writes the
    ordered selection partitions (``relevant_requirements`` /
    ``partially_relevant_requirements``) AND
    ``requirement_classifications`` — the explicit DIRECT / PARTIAL /
    NOT_MENTIONED decision for every project requirement, keyed by normalized
    requirement name. Selection is therefore the ONLY producer of the
    relevance classification; ``match_skills_and_evidence`` consumes the
    selection partitions, the evidence understanding, and request data only;
    ``calculate_fit`` consumes ``requirement_classifications`` and the
    matching data and never re-classifies. All analytical fields are written
    from authoritative request data or deterministic classification; the
    deterministic final result is assembled from them.
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