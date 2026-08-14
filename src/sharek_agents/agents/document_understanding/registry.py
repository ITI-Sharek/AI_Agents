from __future__ import annotations

from pydantic import ValidationError

from sharek_agents.agents.document_understanding.tools import (
    NativeToolCall,
    Tool,
    ToolDefinition,
    ToolResult,
    ToolResultStatus,
)


class ToolRegistryError(Exception):
    """Raised when a tool operation cannot be completed."""


def _safe_tool_error(error: Exception) -> str:
    """Produce a safe error message without stack traces or credentials."""
    return f"{type(error).__name__}: {error}"


class ToolRegistry:
    """Request-scoped registry of tools available to the current Agent execution.

    The registry:

    * contains only tools registered for this specific request
    * is never shared between requests (no module-level state)
    * provides deterministic (insertion-ordered) tool listing

    Typical usage::

        registry = ToolRegistry()
        registry.register(search_tool)
        registry.register(inspect_tool)

        # Present tools to the LLM
        definitions = registry.list_definitions()
        model = model.bind_tools([...definitions...])

        # Execute a tool call from the LLM
        result = await registry.execute_call(native_tool_call)
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
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                status="not_found",
                error_message=(
                    f"Tool '{call.name}' is not registered "
                    f"in the current request context"
                ),
            )
        try:
            output = await tool.execute(**call.arguments)
            status: ToolResultStatus = "empty" if not output else "success"
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
        except Exception as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                status="execution_error",
                error_message=_safe_tool_error(exc),
            )
