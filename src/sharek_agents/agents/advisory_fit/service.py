from __future__ import annotations

import asyncio
import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from sharek_agents.agents.advisory_fit.prompts import HUMAN_PROMPT, SYSTEM_PROMPT
from sharek_agents.agents.advisory_fit.schemas import (
    AdvisoryFitAIOutput,
    AdvisoryFitInput,
    AdvisoryFitResult,
    Assessment,
    RequirementAnalysis,
    SkillItem,
)
from sharek_agents.agents.advisory_fit.scoring import (
    calculate_fit_percentage,
    calculate_level_match,
)
from sharek_agents.common.llm import get_llm
from sharek_agents.config import settings


logger = logging.getLogger(__name__)

PROMPT_VERSION = "advisory-fit-v2"
SCHEMA_VERSION = "advisory-fit-result-v1"


class AdvisoryFitProviderError(Exception):
    pass


class AdvisoryFitProviderTimeout(AdvisoryFitProviderError):
    pass


def _build_contributor_level_map(
    skills: list[SkillItem],
) -> dict[str, str]:
    return {s.skill.casefold(): s.level for s in skills}


def _build_assessment(
    required: SkillItem,
    analysis: RequirementAnalysis,
    contributor_level_map: dict[str, str],
) -> Assessment:
    contributor_level = contributor_level_map.get(required.skill.casefold())
    level_match = calculate_level_match(required.level, contributor_level)
    skill_match = analysis.skill_match
    if (
        skill_match == "MATCHED"
        and required.skill.casefold() not in contributor_level_map
    ):
        skill_match = "NOT_EVIDENCED"
    return Assessment(
        skill=required.skill,
        required_level=required.level,
        contributor_level=contributor_level,
        skill_match=skill_match,
        level_match=level_match,
        approach_relevance=analysis.approach_relevance,
        explanation=analysis.explanation,
    )


def _validate_ai_coverage(
    ai_output: AdvisoryFitAIOutput,
    input_data: AdvisoryFitInput,
) -> None:
    input_skills = {r.skill.casefold() for r in input_data.project_requirements}
    ai_skills = {a.skill.casefold() for a in ai_output.assessments}

    if ai_skills != input_skills:
        missing = input_skills - ai_skills
        extra = ai_skills - input_skills
        parts: list[str] = []
        if missing:
            parts.append(f"missing requirements: {sorted(missing)}")
        if extra:
            parts.append(f"unexpected requirements: {sorted(extra)}")
        raise AdvisoryFitProviderError(
            f"AI analysis does not match input requirements; {'; '.join(parts)}"
        )


async def _invoke_analysis_llm(
    input_data: AdvisoryFitInput,
) -> AdvisoryFitAIOutput:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )

    structured = get_llm().with_structured_output(AdvisoryFitAIOutput)
    result = await asyncio.wait_for(
        (prompt | structured).ainvoke(
            {
                "project_requirements": json.dumps(
                    [
                        {"skill": r.skill, "level": r.level}
                        for r in input_data.project_requirements
                    ],
                    indent=2,
                ),
                "contributor_skills": json.dumps(
                    [
                        {"skill": s.skill, "level": s.level}
                        for s in input_data.contributor_skills
                    ],
                    indent=2,
                ),
                "contributor_approach": input_data.contributor_approach,
            }
        ),
        timeout=settings.ai_skill_profile_timeout_seconds,
    )
    return AdvisoryFitAIOutput.model_validate(result)


async def generate_advisory_fit(
    input_data: AdvisoryFitInput,
) -> AdvisoryFitResult:
    logger.info(
        "Starting Advisory Fit analysis: %d requirements, %d contributor skills",
        len(input_data.project_requirements),
        len(input_data.contributor_skills),
    )

    contributor_level_map = _build_contributor_level_map(
        input_data.contributor_skills
    )

    try:
        ai_output = await _invoke_analysis_llm(input_data)
    except asyncio.TimeoutError as exc:
        raise AdvisoryFitProviderTimeout(
            "Advisory Fit provider timed out"
        ) from exc
    except ValidationError as exc:
        raise AdvisoryFitProviderError(
            f"Advisory Fit provider returned invalid output: {exc}"
        ) from exc
    except Exception as exc:
        raise AdvisoryFitProviderError(
            f"Advisory Fit provider failed: {exc}"
        ) from exc

    _validate_ai_coverage(ai_output, input_data)

    ai_by_skill: dict[str, RequirementAnalysis] = {}
    for analysis in ai_output.assessments:
        ai_by_skill[analysis.skill.casefold()] = analysis

    assessments = [
        _build_assessment(req, ai_by_skill[req.skill.casefold()], contributor_level_map)
        for req in input_data.project_requirements
    ]

    fit_percentage = calculate_fit_percentage(assessments)

    total_requirements = len(assessments)
    matched = sum(
        1 for a in assessments if a.skill_match == "MATCHED"
    )
    exact_levels = sum(
        1 for a in assessments if a.level_match == "EXACT"
    )
    summary = (
        f"Advisory Fit assessment for {total_requirements} project "
        f"requirements: {matched} skills matched, {exact_levels} exact level "
        f"matches, fit percentage {fit_percentage}%."
    )

    logger.info(
        "Advisory Fit complete: fit_percentage=%.2f, requirements=%d, matched=%d",
        fit_percentage,
        total_requirements,
        matched,
    )

    return AdvisoryFitResult(
        fit_percentage=fit_percentage,
        assessments=assessments,
        summary=summary,
    )
