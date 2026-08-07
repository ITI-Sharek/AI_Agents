from __future__ import annotations

import asyncio
import logging

from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from sharek_agents.agents.advisory_fit import service as service_module
from sharek_agents.agents.advisory_fit.schemas import (
    AdvisoryFitInput,
    AdvisoryFitResult,
)
from sharek_agents.agents.advisory_fit.state import AgentState
from sharek_agents.agents.advisory_fit.workflow import build_workflow_graph


logger = logging.getLogger(__name__)


graph: CompiledStateGraph = build_workflow_graph().compile()


async def run_advisory_fit_agent(
    input_data: AdvisoryFitInput,
) -> AdvisoryFitResult:
    """Execute the bounded LangGraph advisory-fit workflow.

    Linear workflow: ``initialize_state → understand_approach →
    select_relevant_requirements → match_skills_and_evidence → calculate_fit``.
    Provider exception types and messages are preserved.
    """
    initial_state: AgentState = {"request": input_data}
    try:
        final_state = await graph.ainvoke(initial_state)
    except asyncio.TimeoutError as exc:
        raise service_module.AdvisoryFitProviderTimeout(
            "Advisory Fit provider timed out"
        ) from exc
    except ValidationError as exc:
        raise service_module.AdvisoryFitProviderError(
            f"Advisory Fit provider returned invalid output: {exc}"
        ) from exc
    except (
        service_module.AdvisoryFitProviderError,
        service_module.AdvisoryFitProviderTimeout,
    ) as exc:
        raise exc
    except Exception as exc:
        raise service_module.AdvisoryFitProviderError(
            f"Advisory Fit provider failed: {exc}"
        ) from exc
    return final_state["final_result"]