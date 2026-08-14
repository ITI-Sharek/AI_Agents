"""Tool abstractions for the Skill Profiling Agent (Phase 2, prepared for Phase 7).

A small, request-scoped ``ToolRegistry`` plus the single deterministic
test tool ``get_agent_context``. MCP-backed tools are exposed through
``McpToolAdapter`` — thin adapters around MCP client calls implementing
the same ``Tool`` protocol; no MCP server tools are defined here.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError, model_validator

from sharek_agents.agents.skill_profiling.contract_schemas import SkillProfileInput
from sharek_agents.agents.skill_profiling_agent.detection.detector import (
    FrameworksDetector,
)
from sharek_agents.agents.skill_profiling_agent.detection.github import (
    GitHubClient,
    RepositoryContext,
)
from sharek_agents.agents.skill_profiling_agent.mcp_client import (
    MCPClient,
    MCPToolDefinition,
)


class ToolDefinition(BaseModel):
    """JSON Schema describing a callable tool for LLM native tool calling."""

    name: str = Field(description="Unique tool name")
    description: str = Field(description="What the tool does")
    parameters: dict[str, Any] = Field(
        description="JSON Schema object describing valid arguments",
    )


class NativeToolCall(BaseModel):
    """A structured tool call returned natively by the LLM."""

    id: str = Field(description="Provider-assigned tool call identifier")
    name: str = Field(description="Name of the tool to invoke")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments parsed by the provider from the LLM response",
    )


class ToolResult(BaseModel):
    """Result of executing a single ``NativeToolCall``.

    The ``status`` field distinguishes execution outcomes so callers can
    handle failures without inspecting exception types.
    """

    tool_call_id: str = Field(default="", description="Matches NativeToolCall.id")
    name: str = Field(default="", description="Tool name that was executed")
    status: Literal[
        "success", "validation_error", "execution_error", "not_found", "empty"
    ] = Field(default="success", description="Execution outcome")
    output: str = Field(
        default="",
        description="String-serialised result returned to the LLM",
    )
    error_message: str = Field(
        default="",
        description="Safe, non-technical error description",
    )


class Tool(Protocol):
    """A callable tool with metadata for native LLM tool calling."""

    @property
    def name(self) -> str:
        """Short unique identifier, must match ToolDefinition.name."""
        ...

    @property
    def definition(self) -> ToolDefinition:
        """Full JSON Schema definition for the LLM provider."""
        ...

    async def execute(self, **kwargs: Any) -> str:
        """Execute the tool and return a string result for the LLM."""
        ...


class ToolRegistryError(Exception):
    """Raised when a tool cannot be registered."""


class RepoAuthorizationError(Exception):
    """The requested repository is not authorized by this request."""


class ToolRegistry:
    """Request-scoped registry of tools available to one agent execution.

    * contains only tools registered for this specific request,
    * is never shared between requests (no module-level state),
    * provides deterministic (insertion-ordered) tool listing.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolRegistryError(
                f"Tool '{tool.name}' is already registered",
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def list_definitions(self) -> list[ToolDefinition]:
        return [tool.definition for tool in self._tools.values()]

    @property
    def size(self) -> int:
        return len(self._tools)

    async def execute_call(self, call: NativeToolCall) -> ToolResult:
        """Execute one tool call, converting any failure into a ToolResult.

        Unknown tools and raised exceptions are returned as structured
        ``ToolResult`` objects (never propagated) so the agent can feed
        the failure back to the LLM safely.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                status="not_found",
                error_message=f"Tool '{call.name}' is not registered",
            )
        try:
            output = await tool.execute(**call.arguments)
            status: str = "empty" if not output else "success"
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                status=status,
                output=output,
            )
        except ValidationError as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                status="validation_error",
                error_message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - converted to ToolResult by design
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                status="execution_error",
                error_message=f"{type(exc).__name__}: {exc}",
            )


class McpToolAdapter:
    """Thin ``Tool`` adapter exposing one MCP server tool to the agent.

    The adapter only translates: an agent tool call becomes an MCP
    ``tools/call`` through the client, and the normalized structured
    result is serialized back for the LLM. It performs no analysis and
    contains no MCP server logic. Failures raised by the client
    (``MCPClientError`` subclasses) propagate to the caller, where the
    agent's registry converts them into safe ``ToolResult`` records.

    The request-scoped ``github_pat`` may be supplied at construction.
    When the wrapped tool's definition declares a ``github_pat``
    argument (e.g. the orchestration tool
    ``analyze_contributor_repository``), the adapter injects that PAT
    into the call arguments before the MCP ``tools/call``. The LLM never
    sees the PAT: it is not part of prompts, tool definitions, results,
    evidence, logs, or error messages — it travels only inside the
    structured call arguments sent to the MCP server for repository
    acquisition. The request PAT is the ONLY allowed PAT source: any
    LLM-supplied ``github_pat`` value is replaced by the request PAT, and
    without a request PAT it is dropped (``None`` is sent instead), so
    the LLM can never introduce a credential and public repositories keep
    working unchanged.

    Repository authorization: tools that declare a ``repo_url`` argument
    (the repository-analysis orchestration tool) are restricted to the
    request's authorized repositories (the canonical ``html_url`` values
    of ``selected_repositories``). Any other repository URL fails closed
    BEFORE the PAT is injected and BEFORE the MCP ``tools/call``, so an
    LLM-generated URL can never expand the request's authorization scope
    and the request PAT is never sent for an unauthorized repository.

    Contributor/project mode (Phase 26): the request-scoped
    ``github_login`` selects the MCP analysis mode for tools declaring a
    ``contributor_identifier`` argument (e.g. the orchestration tool
    ``analyze_contributor_repository``). When the request carries a
    GitHub login (CONTRIBUTOR analysis), that exact login is injected as
    the ``contributor_identifier`` — the LLM may never supply a
    different one. When the request carries no GitHub login (PROJECT
    analysis), any LLM-supplied ``contributor_identifier`` is dropped so
    the MCP server enters PROJECT mode: the LLM can never invent a
    contributor identity.
    """

    def __init__(
        self,
        client: MCPClient,
        definition: MCPToolDefinition,
        *,
        github_pat: str | None = None,
        authorized_repo_urls: list[str] | None = None,
        github_login: str | None = None,
    ) -> None:
        self._client = client
        self._definition = definition
        self._github_pat = github_pat
        self._github_login = (github_login or "").strip()
        self._authorized_repo_urls = (
            frozenset(_canonical_repo_url(url) for url in authorized_repo_urls)
            if authorized_repo_urls is not None
            else None
        )

    @property
    def name(self) -> str:
        return self._definition.name

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._definition.name,
            description=self._definition.description,
            parameters=self._definition.parameters,
        )

    async def execute(self, **kwargs: Any) -> str:
        arguments = self._authorize_repo_url(kwargs)
        arguments = self._inject_github_pat(arguments)
        arguments = self._inject_contributor_identifier(arguments)
        result = await self._client.call_tool(self.name, arguments)
        return result.to_text()

    def _declares_argument(self, name: str) -> bool:
        """Whether the wrapped tool's schema declares an argument ``name``."""
        properties = self._definition.parameters.get("properties")
        return isinstance(properties, dict) and name in properties

    def _authorize_repo_url(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Restrict repository-analysis tools to the request's authorized repositories.

        For tools declaring a ``repo_url`` argument (e.g.
        ``analyze_contributor_repository``), the supplied URL must be one
        of the request's authorized repositories (canonical
        ``selected_repositories`` ``html_url`` identity). An unauthorized,
        missing, or non-string URL fails closed with
        ``RepoAuthorizationError`` BEFORE any PAT injection and BEFORE the
        MCP ``tools/call``, so the request PAT can never reach the server
        for a repository outside the request scope. Tools without a
        ``repo_url`` argument are unaffected.
        """
        if not self._declares_argument("repo_url"):
            return arguments
        repo_url = arguments.get("repo_url")
        if not isinstance(repo_url, str) or not repo_url.strip():
            raise RepoAuthorizationError(
                "repository URL is required and must be one of the "
                "request's selected repositories"
            )
        if self._authorized_repo_urls is None:
            raise RepoAuthorizationError(
                "repository analysis is restricted to the request's "
                "selected repositories"
            )
        if _canonical_repo_url(repo_url) not in self._authorized_repo_urls:
            raise RepoAuthorizationError(
                f"repository '{repo_url}' is not authorized by this "
                "request; analyze_contributor_repository may only be "
                "called with one of the request's selected repositories"
            )
        return arguments

    def _inject_github_pat(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Normalize the ``github_pat`` argument to the request PAT ONLY.

        ``request.github_pat`` is the only allowed PAT source. When the
        wrapped tool declares a ``github_pat`` argument (e.g.
        ``analyze_contributor_repository``), a request PAT replaces any
        LLM-supplied value (the LLM never knows the real PAT); with no
        request PAT any LLM-supplied value is dropped (``None`` is sent
        instead), so the LLM can never introduce a credential. Tools
        without a ``github_pat`` argument are unaffected. Only called
        after ``_authorize_repo_url``, so this runs solely for
        repositories authorized by the request.
        """
        if not self._declares_argument("github_pat"):
            return arguments
        if self._github_pat is not None:
            return {**arguments, "github_pat": self._github_pat}
        if "github_pat" in arguments:
            return {**arguments, "github_pat": None}
        return arguments

    def _inject_contributor_identifier(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Normalize the ``contributor_identifier`` argument to the request GitHub login ONLY.

        ``request.github_login`` is the only allowed contributor
        identity. When the wrapped tool declares a
        ``contributor_identifier`` argument (e.g.
        ``analyze_contributor_repository``): a request GitHub login
        (CONTRIBUTOR analysis) replaces any LLM-supplied value, so the
        LLM can never attribute the analysis to a different identity;
        without a request GitHub login (PROJECT analysis) any
        LLM-supplied value is dropped (``None`` is sent instead), so
        the LLM can never invent a contributor identity and the MCP
        server enters PROJECT mode. Only called after
        ``_authorize_repo_url`` and ``_inject_github_pat``.
        """
        if not self._declares_argument("contributor_identifier"):
            return arguments
        if self._github_login:
            return {**arguments, "contributor_identifier": self._github_login}
        if "contributor_identifier" in arguments:
            return {**arguments, "contributor_identifier": None}
        return arguments


def build_mcp_tool_adapters(
    client: MCPClient,
    definitions: list[MCPToolDefinition],
    *,
    github_pat: str | None = None,
    authorized_repo_urls: list[str] | None = None,
    github_login: str | None = None,
) -> list[McpToolAdapter]:
    """Build one thin adapter per server-reported tool definition.

    ``github_pat`` is the request-scoped credential injected by the
    adapters into calls to tools that declare a ``github_pat`` argument
    (e.g. ``analyze_contributor_repository``); it is never exposed to
    the LLM. ``None`` (no PAT in the request) injects nothing.

    ``github_login`` is the request-scoped contributor identity
    (``SkillProfileInput.github_login``) that selects the MCP
    analysis mode: a non-empty login is injected as the
    ``contributor_identifier`` argument (CONTRIBUTOR analysis), while an
    empty login causes any LLM-supplied ``contributor_identifier`` to be
    dropped (PROJECT analysis). The LLM never chooses the identity.

    ``authorized_repo_urls`` are the canonical repository URLs authorized
    by the request (the ``html_url`` values of ``selected_repositories``).
    Adapters for repository-analysis tools (those declaring a ``repo_url``
    argument) reject any repository URL outside this set before the MCP
    call, so an LLM-generated URL can never expand the request's
    authorization scope or trigger PAT injection for an unauthorized
    repository.
    """
    return [
        McpToolAdapter(
            client,
            definition,
            github_pat=github_pat,
            authorized_repo_urls=authorized_repo_urls,
            github_login=github_login,
        )
        for definition in definitions
    ]


def _canonical_repo_url(url: str) -> str:
    """Normalize a repository URL to its canonical identity for comparison.

    Strips surrounding whitespace and trailing slashes, folds the scheme
    to ``https`` and the host to lowercase (removing an optional ``www.``
    host prefix), so safe equivalent GitHub URL forms compare equal while
    the repository path stays exact. A different host, owner, or
    repository never canonicalizes to an authorized URL.
    """
    normalized = url.strip().rstrip("/")
    if "://" not in normalized:
        return normalized
    scheme, remainder = normalized.split("://", 1)
    host, _, path = remainder.partition("/")
    if host.lower().startswith("www."):
        host = host[4:]
    return f"{'https' if scheme.lower() == 'http' else scheme.lower()}://{host.lower()}/{path}"


class GetAgentContextTool:
    """Deterministic test tool returning a compact summary of the request.

    The output never includes the request-scoped ``github_pat`` or any
    other credential material.
    """

    def __init__(self, request: SkillProfileInput) -> None:
        self._request = request

    @property
    def name(self) -> str:
        return "get_agent_context"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Return a compact summary of the skill-profiling request "
                "context: generation ID, contributor ID, GitHub login, "
                "analysis mode (CONTRIBUTOR when a GitHub login is "
                "present, otherwise PROJECT), role, and the selected "
                "repositories with their canonical HTML URL, primary "
                "language, and technologies."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )

    async def execute(self, **kwargs: Any) -> str:
        request = self._request
        context = {
            "generation_id": request.generation_id,
            "contributor_id": request.contributor_id,
            "github_login": request.github_login,
            "analysis_mode": "CONTRIBUTOR" if request.github_login.strip() else "PROJECT",
            "role": request.role,
            "repository_count": len(request.selected_repositories),
            "repositories": [
                {
                    "full_name": repo.full_name,
                    "html_url": repo.html_url,
                    "primary_language": repo.primary_language,
                    "technologies": repo.technologies,
                }
                for repo in request.selected_repositories
            ],
        }
        return json.dumps(context, ensure_ascii=False)


class DetectFrameworksArgs(BaseModel):
    """Arguments accepted by ``DetectFrameworksTool``."""

    repository: str = Field(
        min_length=1,
        description="GitHub repository identifier in 'owner/name' form",
    )
    branch: str | None = Field(
        default=None,
        description=(
            "Optional branch or ref to analyze; defaults to the "
            "repository's default branch"
        ),
    )

    @model_validator(mode="after")
    def repository_must_be_owner_name(self) -> DetectFrameworksArgs:
        parts = self.repository.strip().strip("/").split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                "repository must be an 'owner/name' GitHub repository identifier"
            )
        return self


class DetectFrameworksTool:
    """ReAct agent tool detecting technologies from dependency manifests.

    Fetches a repository's dependency files through the GitHub API
    (never cloning), parses them in memory, and matches package names
    against a static technology registry. The tool performs no LLM
    reasoning and is fully deterministic.

    The GitHub token travels through the tool's request context and is
    never included in output, logs, or error messages.
    """

    def __init__(
        self,
        context: RepositoryContext,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._context = context
        self._http_client = http_client
        self._detector: FrameworksDetector | None = None

    @property
    def name(self) -> str:
        return "detect_frameworks"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Detect frameworks, ORMs, template engines, testing tools "
                "and libraries used in a GitHub repository by reading its "
                "dependency manifest files through the GitHub API. Call "
                "with a repository in 'owner/name' form; returns structured "
                "detection evidence with source files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "repository": {
                        "type": "string",
                        "description": (
                            "GitHub repository identifier in 'owner/name' form"
                        ),
                    },
                    "branch": {
                        "type": ["string", "null"],
                        "description": (
                            "Optional branch or ref to analyze; defaults to "
                            "the repository's default branch"
                        ),
                    },
                },
                "required": ["repository"],
                "additionalProperties": False,
            },
        )

    def _get_detector(self) -> FrameworksDetector:
        if self._detector is None:
            client = (
                GitHubClient(self._context.github_token, http_client=self._http_client)
                if self._http_client is not None
                else None
            )
            self._detector = FrameworksDetector(
                context=self._context,
                client=client,
            )
        return self._detector

    async def execute(self, **kwargs: Any) -> str:
        args = DetectFrameworksArgs.model_validate(kwargs)
        detector = self._get_detector()
        report = await detector.detect(args.repository, branch=args.branch)
        return report.to_json()
