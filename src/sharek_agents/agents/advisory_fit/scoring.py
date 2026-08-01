from __future__ import annotations

from sharek_agents.agents.advisory_fit.schemas import (
    Assessment,
    LevelMatch,
    SkillLevel,
)


SKILL_MATCH_SCORES: dict[str, float] = {
    "MATCHED": 100.0,
    "NOT_MATCHED": 0.0,
    "NOT_EVIDENCED": 0.0,
}

LEVEL_MATCH_SCORES: dict[str, float] = {
    "EXACT": 100.0,
    "HIGHER": 100.0,
    "LOWER": 50.0,
    "MISSING": 0.0,
}

APPROACH_RELEVANCE_SCORES: dict[str, float] = {
    "DIRECT": 100.0,
    "PARTIAL": 50.0,
    "NOT_MENTIONED": 0.0,
    "UNCLEAR": 25.0,
}

SKILL_MATCH_WEIGHT: float = 0.40
LEVEL_MATCH_WEIGHT: float = 0.40
APPROACH_RELEVANCE_WEIGHT: float = 0.20

_LEVEL_ORDER: dict[str, int] = {
    "beginner": 0,
    "intermediate": 1,
    "advanced": 2,
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
    level_score = LEVEL_MATCH_SCORES.get(assessment.level_match, 0.0)
    approach_score = APPROACH_RELEVANCE_SCORES.get(
        assessment.approach_relevance, 0.0
    )
    return (
        skill_score * SKILL_MATCH_WEIGHT
        + level_score * LEVEL_MATCH_WEIGHT
        + approach_score * APPROACH_RELEVANCE_WEIGHT
    )


def calculate_fit_percentage(assessments: list[Assessment]) -> float:
    if not assessments:
        return 0.0
    total = sum(_requirement_score(a) for a in assessments)
    return round(total / len(assessments), 2)
