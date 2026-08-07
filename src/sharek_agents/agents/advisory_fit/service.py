from __future__ import annotations

import json
import logging

from sharek_agents.agents.advisory_fit.schemas import (
    AdvisoryFitInput,
    AdvisoryFitResult,
    Evidence,
)


logger = logging.getLogger(__name__)


class AdvisoryFitProviderError(Exception):
    pass


class AdvisoryFitProviderTimeout(AdvisoryFitProviderError):
    pass


def _serialize_evidence(evidence: list[Evidence]) -> str:
    """Serialize contributor evidence as opaque JSON for prompt embedding."""
    return json.dumps(
        [e.model_dump(mode="json") for e in evidence],
        indent=2,
    )


async def generate_advisory_fit(
    input_data: AdvisoryFitInput,
) -> AdvisoryFitResult:
    logger.info(
        "Starting Advisory Fit analysis: %d requirements, %d contributor skills",
        len(input_data.project_requirements),
        len(input_data.contributor.skills),
    )

    result = await _get_graph_runner()(input_data)

    logger.info(
        "Advisory Fit complete: fit_percentage=%.2f, requirements=%d, matched=%d",
        result.fit_percentage,
        len(result.assessments),
        sum(1 for a in result.assessments if a.skill_match == "MATCHED"),
    )

    return result


def _get_graph_runner():
    """Return the LangGraph agent runner, importing it lazily.

    The graph module imports this service module (for the shared exception
    types), so the import is deferred to break the module cycle.
    """
    from sharek_agents.agents.advisory_fit.graph import run_advisory_fit_agent

    return run_advisory_fit_agent