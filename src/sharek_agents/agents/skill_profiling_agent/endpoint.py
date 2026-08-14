"""FastAPI endpoint handler for the new Skill Profiling Agent (Phase 1).

The endpoint handler is registered in ``main.py`` with the existing
service authentication mechanism (``require_service_token``) and the
existing input contract (``SkillProfileInput``). It returns a clearly
defined temporary response and does not perform any LLM or external
analysis calls yet.

Phase 16: the endpoint resolves an injectable MCP configuration and
passes it to the service. Configuration is read from the environment
(``SKILL_PROFILING_MCP_ENDPOINT_URL`` and optionally
``SKILL_PROFILING_MCP_BEARER_TOKEN`` /
``SKILL_PROFILING_MCP_TIMEOUT_SECONDS``). Callers may also pass an
explicit ``mcp_config`` for direct injection. No environment file is
edited by this package.

Since Phase 23B MCP configuration is REQUIRED: when no endpoint URL is
configured, resolution yields ``None`` and the service fails the
request with a clear MCP configuration error — the agent never silently
runs local-only.
"""

from __future__ import annotations

import os

from fastapi import HTTPException, status

from sharek_agents.agents.skill_profiling.contract_schemas import SkillProfileInput
from sharek_agents.agents.skill_profiling_agent.mcp_client import (
    MCPClientConfig,
    MCP_PIPELINE_MAX_SECONDS,
)
from sharek_agents.agents.skill_profiling_agent.service import (
    SkillProfileAgentError,
    SkillProfileAgentTimeout,
    generate_skill_profile_agent,
)
from sharek_agents.common.logging import get_logger

logger = get_logger(__name__)

ENV_MCP_ENDPOINT_URL = "SKILL_PROFILING_MCP_ENDPOINT_URL"
ENV_MCP_BEARER_TOKEN = "SKILL_PROFILING_MCP_BEARER_TOKEN"
ENV_MCP_TIMEOUT_SECONDS = "SKILL_PROFILING_MCP_TIMEOUT_SECONDS"

_DEFAULT_MCP_TIMEOUT_SECONDS = MCP_PIPELINE_MAX_SECONDS


def resolve_mcp_config() -> MCPClientConfig | None:
    """Resolve the production MCP configuration from the environment.

    Returns ``None`` when no ``SKILL_PROFILING_MCP_ENDPOINT_URL`` is
    configured. MCP is REQUIRED for the Skill Profiling Agent: the
    service treats ``None`` as a configuration error and fails the
    request, so a missing endpoint URL never silently enables local-only
    mode. The bearer token is optional; when present it is sent only in
    the MCP Authorization header and is never logged.
    """
    endpoint_url = os.environ.get(ENV_MCP_ENDPOINT_URL, "").strip()
    if not endpoint_url:
        return None

    try:
        timeout_seconds = float(
            os.environ.get(ENV_MCP_TIMEOUT_SECONDS, "") or _DEFAULT_MCP_TIMEOUT_SECONDS
        )
    except ValueError:
        timeout_seconds = _DEFAULT_MCP_TIMEOUT_SECONDS

    return MCPClientConfig(
        endpoint_url=endpoint_url,
        bearer_token=os.environ.get(ENV_MCP_BEARER_TOKEN) or None,
        timeout_seconds=timeout_seconds,
    )


async def generate_skill_profile_agent_endpoint(
    body: SkillProfileInput,
    mcp_config: MCPClientConfig | None = None,
):
    """Generate a skill profile with the new Skill Profiling Agent.

    Phase 1: validates the request and returns the temporary agent
    response without invoking any LLM or external analysis.

    Phase 16: when no ``mcp_config`` is passed explicitly, the endpoint
    resolves one from the environment and passes it to the service, so
    production can enable MCP tool discovery through ``tools/list``
    without code changes.

    Phase 23B: MCP configuration is REQUIRED. When resolution yields
    ``None`` (no endpoint URL configured), the service fails with a
    clear ``SkillProfileAgentError``, surfaced here as a 502 — the
    agent never runs local-only.
    """
    if mcp_config is None:
        mcp_config = resolve_mcp_config()
    try:
        return await generate_skill_profile_agent(body, mcp_config=mcp_config)
    except SkillProfileAgentTimeout as exc:
        logger.warning("Skill Profiling Agent timed out: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Skill Profiling Agent timed out: {exc}",
        ) from exc
    except SkillProfileAgentError as exc:
        logger.warning("Skill Profiling Agent error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Skill Profiling Agent error: {exc}",
        ) from exc
