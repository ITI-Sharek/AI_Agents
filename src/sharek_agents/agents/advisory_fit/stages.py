from __future__ import annotations

from sharek_agents.agents.advisory_fit.schemas import Assessment


def _build_summary(
    assessments: list[Assessment],
    fit_percentage: float,
) -> str:
    total_requirements = len(assessments)
    matched = sum(1 for a in assessments if a.skill_match == "MATCHED")
    evidenced = sum(
        1 for a in assessments if a.evidence_match != "MISSING"
    )
    exact_levels = sum(1 for a in assessments if a.level_match == "EXACT")
    return (
        f"Advisory Fit assessment for {total_requirements} project "
        f"requirements: {matched} skills matched, {evidenced} requirements "
        f"evidence-supported, {exact_levels} exact level matches, fit "
        f"percentage {fit_percentage}%."
    )