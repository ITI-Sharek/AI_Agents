"""Orchestration entry point for the new Skill Profiling Agent.

Phase 2: builds a request-scoped ToolRegistry with the deterministic
``get_agent_context`` tool, runs the ReAct agent core, and maps agent
failures to the service-level errors consumed by the endpoint.

Phase 10: the service discovers the real MCP server's tools via
``tools/list``, wraps them in thin ``McpToolAdapter`` instances, and
registers them in the same request-scoped ToolRegistry so the ReAct
agent's LLM can select and call them through the existing loop. The MCP
tool list is never hard-coded: the server remains the source of truth.

Since Phase 23B MCP configuration is REQUIRED: a missing
``MCPClientConfig`` (no ``SKILL_PROFILING_MCP_ENDPOINT_URL`` configured)
is a service error, never a silent local-only run.

Phase 16: the request-scoped local tools now include the deterministic
``DetectFrameworksTool``, wired with the request's repository capsules
(default branches) and its request-scoped ``github_pat``. The PAT is
used only inside the tool's GitHub client; it never appears in tool
definitions, outputs, error messages, logs, or evidence. The detection
tool is registered only when the request actually carries a ``github_pat``
(the tool cannot authenticate without one); otherwise only
``get_agent_context`` is registered among the local tools (MCP tools are
always registered). Tool registration stays additive and
order-preserving, and the ReAct loop remains the only decision maker.

Phase 24: the same request-scoped ``github_pat`` is handed to the MCP
tool adapters (``build_mcp_tool_adapters``). The adapters inject it as
the ``github_pat`` argument of MCP tools that declare it (the
orchestration tool ``analyze_contributor_repository``), so the PAT
reaches repository acquisition through ``tools/call`` without ever being
exposed to the LLM prompt, evidence, results, logs, or responses. The
adapters also receive the request's authorized repository URLs (the
canonical ``html_url`` values of ``selected_repositories``): repository-
analysis tools reject any other ``repo_url`` before the PAT is injected
or sent, so an LLM-generated repository URL can never expand the
request's authorization scope.

Phase 26: the Agent is now mode-aware. The existing request data
carries the contributor/project distinction: a non-empty
``request.github_login`` means CONTRIBUTOR profiling — the adapters
inject that exact login as the MCP ``contributor_identifier`` so the
server runs its CONTRIBUTOR flow; an empty ``github_login`` means
PROJECT profiling — the adapters drop any LLM-supplied
``contributor_identifier`` so the server runs its PROJECT flow. No new
request field was introduced.
"""

from __future__ import annotations

import logging

from sharek_agents.agents.skill_profiling.contract_schemas import SkillProfileInput
from sharek_agents.agents.skill_profiling_agent.agent import (
    AgentConfig,
    AgentProviderError,
    AgentTimeoutError,
    SkillProfilingAgent,
)
from sharek_agents.agents.skill_profiling_agent.detection.github import (
    RepositoryContext,
)
from sharek_agents.agents.skill_profiling_agent.mcp_client import (
    MCPClientConfig,
    MCPClientError,
    SkillProfilingMCPClient,
)
from sharek_agents.agents.skill_profiling_agent.schemas import SkillProfileAgentResponse
from sharek_agents.agents.skill_profiling_agent.tools import (
    DetectFrameworksTool,
    GetAgentContextTool,
    build_mcp_tool_adapters,
)

logger = logging.getLogger(__name__)


class SkillProfileAgentError(Exception):
    """Base error for the Skill Profiling Agent service."""


class SkillProfileAgentTimeout(SkillProfileAgentError):
    """The agent exceeded its execution budget."""


def _repository_context(request: SkillProfileInput) -> RepositoryContext:
    """Build the detection tool's request-scoped GitHub context.

    The context carries the request's ``github_pat`` (never logged) and
    the default branches already known from the repository evidence
    capsules, so the detector can skip the repository-metadata API call.
    """
    return RepositoryContext(
        github_token=request.github_pat or "",
        default_branches={
            repository.full_name: repository.default_branch
            for repository in request.selected_repositories
        },
    )


def _local_tools(request: SkillProfileInput) -> list:
    """Build the deterministic local Agent Tools for one request.

    ``get_agent_context`` is always registered. ``detect_frameworks`` is
    registered when the request carries a ``github_pat``, because the
    tool's GitHub API access is authenticated with that PAT and cannot
    function without it. Registration order is deterministic.
    """
    tools = [GetAgentContextTool(request)]
    if request.github_pat:
        tools.append(DetectFrameworksTool(_repository_context(request)))
    else:
        logger.info(
            "No github_pat in request: detect_frameworks tool not registered",
        )
    return tools


async def generate_skill_profile_agent(
    request: SkillProfileInput,
    *,
    mcp_config: MCPClientConfig | None = None,
) -> SkillProfileAgentResponse:
    """Orchestrate the new Skill Profiling Agent (ReAct core + MCP tools).

    MCP is REQUIRED: the service fails with ``SkillProfileAgentError``
    when ``mcp_config`` is missing, so the agent never silently runs
    local-only. With a config, the request-scoped local Agent Tools
    (``get_agent_context`` and, when the request carries a
    ``github_pat``, ``detect_frameworks`` bound to that PAT) are
    registered alongside the MCP server's tools, which are discovered
    dynamically via ``tools/list`` and registered through
    ``McpToolAdapter`` instances; the LLM then decides during the ReAct
    loop which tools to call and when enough evidence has been
    collected. The MCP client is always closed before returning.

    Provider and timeout failures are mapped to ``SkillProfileAgentError``
    / ``SkillProfileAgentTimeout`` so the endpoint can translate them into
    502 / 504 responses. Missing MCP configuration and MCP discovery/
    connection failures are mapped to ``SkillProfileAgentError``;
    tool-call failures stay inside the ReAct loop as safe tool
    observations.
    """
    if mcp_config is None:
        raise SkillProfileAgentError(
            "MCP configuration is required for the Skill Profiling Agent: "
            "set SKILL_PROFILING_MCP_ENDPOINT_URL or pass an explicit "
            "mcp_config"
        )
    logger.info(
        "Skill Profiling Agent invoked: generation=%s contributor=%s repositories=%d",
        request.generation_id,
        request.contributor_id,
        len(request.selected_repositories),
    )

    client: SkillProfilingMCPClient | None = None
    try:
        tools = _local_tools(request)
        client = SkillProfilingMCPClient(mcp_config)
        definitions = await client.list_tools()
        tools.extend(
            build_mcp_tool_adapters(
                client,
                definitions,
                github_pat=request.github_pat,
                github_login=request.github_login,
                authorized_repo_urls=[
                    repository.html_url
                    for repository in request.selected_repositories
                ],
            )
        )
        logger.info(
            "Discovered %d MCP tools via tools/list",
            len(definitions),
        )

        config = AgentConfig(tools=tools)
        agent = SkillProfilingAgent()
        response = await agent.run(request, config=config)
    except AgentTimeoutError as exc:
        raise SkillProfileAgentTimeout(str(exc)) from exc
    except AgentProviderError as exc:
        raise SkillProfileAgentError(str(exc)) from exc
    except MCPClientError as exc:
        raise SkillProfileAgentError(f"MCP tool discovery failed: {exc}") from exc
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:  # noqa: BLE001 - cleanup must never mask the result
                logger.debug("MCP client close failed", exc_info=True)

    logger.info(
        "Skill Profiling Agent responded: status=%s phase=%s iterations=%d tools=%d",
        response.status,
        response.phase,
        response.iterations_used,
        len(response.tool_activities),
    )
    return response
