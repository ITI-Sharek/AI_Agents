"""MCP Streamable HTTP client for the Skill Profiling Agent (Phase 8).

Implements the client-side boundary reserved in Phase 7: a real MCP
client that speaks the MCP protocol over Streamable HTTP against the
``skill_profiling_mcp`` server.

The client is ONLY a communication layer:

* it performs the MCP initialization handshake,
* it discovers tools via ``tools/list``,
* it invokes tools via ``tools/call`` with structured JSON arguments,
* it normalizes MCP tool results into structured client-side models,
* it converts protocol, HTTP, and network failures into the package's
  ``MCPClientError`` exception boundary.

The client does NOT reason, plan, or profile. The ReAct agent remains
the owner of reasoning and tool selection. The client never executes
git, Docker, subprocess, shell, or filesystem commands, never imports
the MCP server implementation, and never logs credentials.

Authentication is optional and configured through ``MCPClientConfig``
(``bearer_token``); the token is sent only in the Authorization header
and is never logged or included in exceptions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from mcp.types import LATEST_PROTOCOL_VERSION
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

JSON_RPC_VERSION = "2.0"
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_SSE = "text/event-stream"
ACCEPT_HEADERS = f"{CONTENT_TYPE_JSON}, {CONTENT_TYPE_SSE}"
MCP_SESSION_ID_HEADER = "mcp-session-id"
MCP_PROTOCOL_VERSION_HEADER = "mcp-protocol-version"
CLIENT_NAME = "sharek-agents-skill-profiling"
CLIENT_VERSION = "0.1.0"
MAX_ERROR_DETAIL = 500

# Worst-case wall-clock duration of one MCP repository-analysis call
# (``analyze_contributor_repository``), derived from the MCP server's own
# per-step timeouts:
#   image build 600s (one-time, cached per fingerprint) + clone 300s
#   + max(filter 600s + scope 120s + static 300s, graphify 300s + select 300s)
#   = 600 + 300 + 1020 = 1920s
MCP_PIPELINE_MAX_SECONDS: float = 1920.0


class MCPClientError(Exception):
    """Base error for the Skill Profiling Agent MCP client boundary."""


class MCPConnectionError(MCPClientError):
    """The MCP client could not establish or maintain a session."""


class MCPToolExecutionError(MCPClientError):
    """An MCP-backed tool call failed while executing on the server."""


class MCPToolDefinition(BaseModel):
    """A tool exposed by the MCP server (source of truth is the server).

    Client-side mirror of the wire tool descriptor; the ReAct agent's
    ``Tool`` protocol consumes a converted version of this in ``tools.py``.
    """

    name: str = Field(description="Tool name as reported by tools/list")
    description: str = Field(default="", description="Server-provided tool description")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema object describing valid arguments",
    )


class MCPToolResult(BaseModel):
    """Normalized, structured result of a single ``tools/call``.

    Raw HTTP and JSON-RPC details never reach the agent; only this model
    (or an ``MCPToolExecutionError``) crosses the client boundary.
    """

    tool_name: str = Field(description="Name of the tool that was invoked")
    content: list[str] = Field(
        default_factory=list,
        description="Text content blocks returned by the server",
    )
    structured_content: dict[str, Any] | None = Field(
        default=None,
        description="Structured content block returned by the server, if any",
    )
    is_error: bool = Field(default=False, description="Server-side isError flag")

    def to_text(self) -> str:
        """Serialize the result into the string handed to the agent."""
        if self.structured_content is not None:
            return json.dumps(self.structured_content, ensure_ascii=False)
        if self.content:
            return "\n".join(self.content)
        return ""


@dataclass(frozen=True)
class MCPClientConfig:
    """Configuration for a single MCP client session.

    The bearer token is optional; when provided it is sent only in the
    ``Authorization`` header. ``__repr__`` masks the token so it never
    leaks into logs or exceptions.
    """

    endpoint_url: str = Field(default="http://localhost:8080/mcp")
    bearer_token: str | None = None
    timeout_seconds: float = MCP_PIPELINE_MAX_SECONDS
    protocol_version: str = LATEST_PROTOCOL_VERSION

    def __repr__(self) -> str:
        token = "<redacted>" if self.bearer_token else None
        return (
            f"MCPClientConfig(endpoint_url={self.endpoint_url!r}, "
            f"bearer_token={token!r}, timeout_seconds={self.timeout_seconds!r}, "
            f"protocol_version={self.protocol_version!r})"
        )


class MCPClient(Protocol):
    """Boundary contract satisfied by the concrete MCP client.

    The ReAct agent and its tool adapters depend only on this surface;
    the transport (Streamable HTTP) stays an implementation detail.
    """

    @property
    def is_initialized(self) -> bool:
        """Whether the MCP session has been established."""
        ...

    async def initialize(self) -> None:
        """Establish the MCP session (idempotent)."""
        ...

    async def list_tools(self) -> list[MCPToolDefinition]:
        """Discover the tools exposed by the MCP server via tools/list."""
        ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Invoke one MCP tool with structured JSON arguments."""
        ...

    async def close(self) -> None:
        """Release the HTTP session (injected clients are not closed)."""
        ...


class SkillProfilingMCPClient:
    """MCP client over Streamable HTTP.

    Implements the protocol flow: ``initialize``, ``notifications/
    initialized``, ``tools/list``, and ``tools/call`` as JSON-RPC 2.0
    messages POSTed to the server endpoint. Supports both SSE and JSON
    response modes as long as the server follows the MCP Streamable HTTP
    semantics.

    An ``httpx.AsyncClient`` may be injected for tests (e.g. with
    ``httpx.MockTransport``); when injected, this client does not close
    it on ``close()``.
    """

    def __init__(
        self,
        config: MCPClientConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._http: httpx.AsyncClient | None = http_client
        self._owns_http = http_client is None
        self._session_id: str | None = None
        self._initialized = False
        self._request_id = 0
        self._base_headers = {
            "Accept": ACCEPT_HEADERS,
            "Content-Type": CONTENT_TYPE_JSON,
            MCP_PROTOCOL_VERSION_HEADER: config.protocol_version,
            "User-Agent": f"{CLIENT_NAME}/{CLIENT_VERSION}",
        }
        if config.bearer_token:
            self._base_headers["Authorization"] = f"Bearer {config.bearer_token}"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def session_id(self) -> str | None:
        """Server-assigned session identifier, when the server is stateful."""
        return self._session_id

    # ── Protocol flow --------------------------------------------------------

    async def initialize(self) -> None:
        """Establish the MCP session (idempotent)."""
        if self._initialized:
            return

        payload: dict[str, Any] = {
            "jsonrpc": JSON_RPC_VERSION,
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": self._config.protocol_version,
                "capabilities": {},
                "clientInfo": {
                    "name": CLIENT_NAME,
                    "version": CLIENT_VERSION,
                },
            },
        }

        try:
            message = await self._request(payload)
            result = self._require_result(message, method="initialize")
        except MCPClientError as exc:
            raise MCPConnectionError(f"MCP initialization failed: {exc}") from exc

        if not isinstance(result, dict) or not isinstance(result.get("protocolVersion"), str):
            raise MCPConnectionError(
                "MCP initialization failed: server returned no protocol version",
            )

        try:
            await self._request(
                {
                    "jsonrpc": JSON_RPC_VERSION,
                    "method": "notifications/initialized",
                    "params": {},
                },
                expect_result=False,
            )
        except MCPClientError as exc:
            raise MCPConnectionError(f"MCP initialization failed: {exc}") from exc

        self._initialized = True
        server_info = result.get("serverInfo")
        server_name = (
            server_info.get("name", "unknown")
            if isinstance(server_info, dict)
            else "unknown"
        )
        logger.info(
            "MCP client initialized: server=%s protocol=%s",
            server_name,
            result["protocolVersion"],
        )

    async def list_tools(self) -> list[MCPToolDefinition]:
        """Discover server tools via ``tools/list`` (auto-initializes)."""
        await self._ensure_initialized()
        message = await self._request(
            {
                "jsonrpc": JSON_RPC_VERSION,
                "id": self._next_id(),
                "method": "tools/list",
                "params": {},
            },
        )
        result = self._require_result(message, method="tools/list")
        raw_tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(raw_tools, list):
            raise MCPClientError(
                "MCP tools/list response is malformed: 'tools' is not a list",
            )

        definitions: list[MCPToolDefinition] = []
        for raw in raw_tools:
            if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
                raise MCPClientError(
                    "MCP tools/list response is malformed: tool entry "
                    "missing a valid 'name'",
                )
            input_schema = raw.get("inputSchema")
            definitions.append(
                MCPToolDefinition(
                    name=raw["name"],
                    description=str(raw.get("description") or ""),
                    parameters=input_schema if isinstance(input_schema, dict) else {},
                ),
            )
        return definitions

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Invoke one MCP tool via ``tools/call`` (auto-initializes).

        Raises ``MCPToolExecutionError`` when the server reports a tool
        failure (JSON-RPC error or ``isError`` result).
        """
        if not isinstance(arguments, dict):
            raise MCPClientError("MCP tool arguments must be a JSON object")
        if not isinstance(name, str) or not name:
            raise MCPClientError("MCP tool name must be a non-empty string")
        await self._ensure_initialized()

        payload: dict[str, Any] = {
            "jsonrpc": JSON_RPC_VERSION,
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }

        try:
            message = await self._request(payload)
            result = self._require_result(message, method="tools/call")
        except MCPClientError as exc:
            raise MCPToolExecutionError(f"MCP tool '{name}' failed: {exc}") from exc

        if not isinstance(result, dict):
            raise MCPToolExecutionError(f"MCP tool '{name}' returned a malformed result")

        normalized = _normalize_tool_result(name, result)
        if normalized.is_error:
            detail = normalized.to_text() or "reported an error"
            raise MCPToolExecutionError(
                f"MCP tool '{name}' returned an error: {detail[:MAX_ERROR_DETAIL]}",
            )
        return normalized

    async def close(self) -> None:
        """Release owned HTTP resources; injected clients stay open."""
        if self._http is not None and self._owns_http:
            await self._http.aclose()
        self._http = None
        self._initialized = False
        self._session_id = None

    async def __aenter__(self) -> SkillProfilingMCPClient:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    # ── Transport ------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _http_client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self._config.timeout_seconds)
            self._owns_http = True
        return self._http

    async def _request(
        self,
        payload: dict[str, Any],
        *,
        expect_result: bool = True,
    ) -> dict[str, Any] | None:
        """POST one JSON-RPC message and return the matching response.

        Raises ``MCPConnectionError`` for network, timeout, and HTTP
        failures; ``MCPClientError`` for malformed or unmatched protocol
        responses.
        """
        client = self._http_client()
        headers = self._base_headers
        if self._session_id:
            headers = {**headers, MCP_SESSION_ID_HEADER: self._session_id}

        try:
            response = await asyncio.wait_for(
                client.post(self._config.endpoint_url, json=payload, headers=headers),
                timeout=self._config.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise MCPConnectionError(
                f"MCP request '{payload.get('method')}' timed out after "
                f"{self._config.timeout_seconds}s",
            ) from exc
        except httpx.RequestError as exc:
            raise MCPConnectionError(
                f"MCP server unreachable: {type(exc).__name__}",
            ) from exc

        session_id = response.headers.get(MCP_SESSION_ID_HEADER)
        if session_id:
            self._session_id = session_id

        logger.debug(
            "MCP HTTP response: method=%s status=%s",
            payload.get("method"),
            response.status_code,
        )

        if response.status_code == 202:
            return None
        if response.status_code != 200:
            self._raise_http_error(payload.get("method"), response)

        return self._parse_response_message(payload, response, expect_result)

    def _raise_http_error(self, method: str | None, response: httpx.Response) -> None:
        detail = ""
        try:
            body = json.loads(response.text)
            error = body.get("error") if isinstance(body, dict) else None
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                detail = error["message"]
        except (json.JSONDecodeError, ValueError):
            pass

        message = f"MCP request '{method}' failed: HTTP {response.status_code}"
        if detail:
            message += f": {detail[:MAX_ERROR_DETAIL]}"
        raise MCPConnectionError(message) from None

    def _parse_response_message(
        self,
        payload: dict[str, Any],
        response: httpx.Response,
        expect_result: bool,
    ) -> dict[str, Any] | None:
        text = response.text
        if not text:
            if expect_result:
                raise MCPClientError("MCP server returned an empty response")
            return None

        content_type = response.headers.get("content-type", "")
        if CONTENT_TYPE_SSE in content_type:
            messages = _parse_sse_messages(text)
        else:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise MCPClientError(
                    "MCP server returned a malformed response: invalid JSON",
                ) from exc
            messages = [parsed] if isinstance(parsed, dict) else []

        if not messages:
            raise MCPClientError(
                "MCP server returned a malformed response: no JSON-RPC message",
            )

        expected_id = payload.get("id")
        for message in messages:
            if message.get("id") == expected_id:
                return message

        raise MCPClientError(
            "MCP server returned a response that does not match the request id",
        )

    @staticmethod
    def _require_result(
        message: dict[str, Any],
        *,
        method: str,
    ) -> Any:
        if "error" in message:
            error = message["error"]
            code = error.get("code") if isinstance(error, dict) else None
            error_message = (
                error.get("message") if isinstance(error, dict) else "unknown error"
            )
            raise MCPClientError(
                f"MCP server error for {method} (code {code}): {error_message}",
            )
        if "result" not in message:
            raise MCPClientError(
                f"MCP {method} response is malformed: missing 'result'",
            )
        return message["result"]


# ── Module-level helpers ------------------------------------------------------


def _normalize_tool_result(name: str, result: dict[str, Any]) -> MCPToolResult:
    """Normalize a raw ``CallToolResult`` dict into ``MCPToolResult``."""
    content: list[str] = []
    raw_content = result.get("content")
    if isinstance(raw_content, list):
        for item in raw_content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                content.append(item["text"])
            elif isinstance(item, dict):
                content.append(json.dumps(item, ensure_ascii=False))
    structured = result.get("structuredContent")
    return MCPToolResult(
        tool_name=name,
        content=content,
        structured_content=structured if isinstance(structured, dict) else None,
        is_error=bool(result.get("isError", False)),
    )


def _parse_sse_messages(text: str) -> list[dict[str, Any]]:
    """Parse ``data:`` lines of an SSE stream into JSON-RPC messages.

    Raises ``MCPClientError`` when a ``data:`` line carries invalid JSON.
    """
    messages: list[dict[str, Any]] = []
    for block in text.split("\n\n"):
        data_lines = [
            line[5:].lstrip()
            for line in block.splitlines()
            if line.startswith("data:")
        ]
        if not data_lines:
            continue
        payload = "\n".join(data_lines)
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise MCPClientError(
                "MCP server returned a malformed SSE response: invalid JSON",
            ) from exc
        if isinstance(parsed, dict):
            messages.append(parsed)
    return messages
