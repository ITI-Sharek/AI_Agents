from __future__ import annotations

from sharek_agents.agents.advisory_fit.state import AgentState


REQUIRED_EVIDENCE = (
    "relevant_requirements",
    "partially_relevant_requirements",
    "requirement_classifications",
    "skill_verification",
    "level_evaluations",
)


def initialize_state(state: AgentState) -> dict:
    """Seed the workflow — a no-op gate node.

    Reads: ``request`` (already present in the initial state).
    Writes: nothing.

    The request is validated by the endpoint contract (``AdvisoryFitInput``)
    before the workflow starts and remains the immutable authoritative
    snapshot; the node exists to satisfy the workflow topology.
    """
    return {}
