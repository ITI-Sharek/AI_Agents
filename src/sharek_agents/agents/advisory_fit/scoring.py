from __future__ import annotations

from sharek_agents.agents.advisory_fit.schemas import (
    Assessment,
    LevelMatch,
    SkillLevel,
)


# Semantic skill-match scores. ``MATCHED`` earns full credit, ``RELATED``
# half credit (a meaningfully related/adjacent skill is useful but is not the
# required skill itself), and ``MISSING`` zero. Relations are classified by
# the LLM and validated by Python; the numeric values are owned by Python
# only — the LLM never produces a score.
SKILL_MATCH_SCORES: dict[str, float] = {
    "MATCHED": 100.0,
    "RELATED": 50.0,
    "MISSING": 0.0,
}

# Semantic evidence-match scores. ``MATCHED`` earns full credit, ``RELATED``
# half credit (evidence of a related capability without direct proof of the
# exact skill), and ``MISSING`` zero. Relations are classified by the LLM and
# validated by Python; the numeric values are owned by Python only.
EVIDENCE_MATCH_SCORES: dict[str, float] = {
    "MATCHED": 100.0,
    "RELATED": 50.0,
    "MISSING": 0.0,
}

LEVEL_MATCH_SCORES: dict[str, float] = {
    "EXACT": 100.0,
    "HIGHER": 100.0,
    # LOWER keeps a partial credit (50% of the level component) by design:
    # the contributor is below the required level, so the requirement is NOT
    # satisfied — ``level_match == "LOWER"`` is the authoritative
    # insufficiency signal — but the percentage reflects partial proximity
    # to the required level.
    "LOWER": 50.0,
    "MISSING": 0.0,
}

APPROACH_RELEVANCE_SCORES: dict[str, float] = {
    "DIRECT": 100.0,
    "PARTIAL": 50.0,
    "NOT_MENTIONED": 0.0,
    "UNCLEAR": 25.0,
}

# Fit weights — must sum to 1.0 (100%). Level (40%) and approach relevance
# (20%) are unchanged from the previous model. The former 40% skill weight
# is split evenly between the semantic skill relation (20%) and the semantic
# evidence relation (20%), so evidence has an explicit contribution while the
# "capability demonstration" component keeps its 40% total share. Both
# relations are classified by the LLM and validated by Python; evidence can
# support a requirement but never replaces the declared skill level, which
# stays authoritative for ``calculate_level_match``.
SKILL_MATCH_WEIGHT: float = 0.20
EVIDENCE_MATCH_WEIGHT: float = 0.20
LEVEL_MATCH_WEIGHT: float = 0.40
APPROACH_RELEVANCE_WEIGHT: float = 0.20

# Authoritative, deterministic level hierarchy:
# beginner(0) < intermediate(1) < advanced(2) < expert(3).
_LEVEL_ORDER: dict[str, int] = {
    "beginner": 0,
    "intermediate": 1,
    "advanced": 2,
    "expert": 3,
}


def calculate_level_match(
    required_level: SkillLevel,
    contributor_level: SkillLevel | None,
) -> LevelMatch:
    if contributor_level is None:
        return "MISSING"
    req = _LEVEL_ORDER[required_level]
    con = _LEVEL_ORDER[contributor_level]
    if con == req:
        return "EXACT"
    if con > req:
        return "HIGHER"
    return "LOWER"


def _requirement_score(assessment: Assessment) -> float:
    skill_score = SKILL_MATCH_SCORES.get(assessment.skill_match, 0.0)
    evidence_score = EVIDENCE_MATCH_SCORES.get(assessment.evidence_match, 0.0)
    level_score = LEVEL_MATCH_SCORES.get(assessment.level_match, 0.0)
    approach_score = APPROACH_RELEVANCE_SCORES.get(
        assessment.approach_relevance, 0.0
    )
    return (
        skill_score * SKILL_MATCH_WEIGHT
        + evidence_score * EVIDENCE_MATCH_WEIGHT
        + level_score * LEVEL_MATCH_WEIGHT
        + approach_score * APPROACH_RELEVANCE_WEIGHT
    )


def calculate_fit_percentage(assessments: list[Assessment]) -> float:
    if not assessments:
        return 0.0
    total = sum(_requirement_score(a) for a in assessments)
    return round(total / len(assessments), 2)
