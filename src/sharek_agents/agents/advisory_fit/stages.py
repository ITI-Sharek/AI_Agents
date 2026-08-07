from __future__ import annotations

from sharek_agents.agents.advisory_fit.schemas import Assessment


def _build_summary(
    assessments: list[Assessment],
    fit_percentage: float,
) -> str:
    total_requirements = len(assessments)
    matched = sum(1 for a in assessments if a.skill_match == "MATCHED")
    exact_levels = sum(1 for a in assessments if a.level_match == "EXACT")
    return (
        f"Advisory Fit assessment for {total_requirements} project "
        f"requirements: {matched} skills matched, {exact_levels} exact level "
        f"matches, fit percentage {fit_percentage}%."
    )