from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from sharek_agents.agents.document_understanding.registry import ToolRegistry
from sharek_agents.agents.document_understanding.tools import (
    NativeToolCall,
    ToolResult,
)


def bind_tools_to_llm(
    model: BaseChatModel,
    registry: ToolRegistry,
    tool_choice: str | None = "auto",
) -> Runnable:
    """Bind tool definitions to a LangChain chat model for native tool calling.

    The returned ``Runnable`` can be invoked like a normal chat model.
    When the LLM decides to call a tool, the response ``AIMessage``
    will contain ``tool_calls`` that can be extracted with
    :func:`extract_tool_calls`.

    Args:
        model: A LangChain ``BaseChatModel`` instance (e.g. ``ChatOpenRouter``).
        registry: The request-scoped ``ToolRegistry`` with registered tools.
        tool_choice: ``"auto"`` (default), ``"any"``, ``"none"``, or a
            specific tool name. Pass ``None`` to use the provider default.

    Returns:
        A ``Runnable`` that accepts the same inputs as the original model.

    Raises:
        RuntimeError: If the model does not support native tool calling.
    """
    if not hasattr(model, "bind_tools"):
        raise RuntimeError(
            f"Model type {type(model).__name__} does not support "
            f"native tool calling (bind_tools not available)",
        )

    definitions = registry.list_definitions()
    if not definitions:
        return model

    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": d.name,
                "description": d.description,
                "parameters": d.parameters,
            },
        }
        for d in definitions
    ]
    kwargs: dict[str, Any] = {}
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    return model.bind_tools(tools, **kwargs)


def extract_tool_calls(message: AIMessage) -> list[NativeToolCall]:
    """Extract native tool calls from an LLM response.

    Args:
        message: The ``AIMessage`` returned by the LLM.

    Returns:
        A list of ``NativeToolCall`` instances.  Empty when the LLM
        chose to respond with text instead of calling a tool.
    """
    if not hasattr(message, "tool_calls") or not message.tool_calls:
        return []

    return [
        NativeToolCall(
            id=tc["id"],
            name=tc["name"],
            arguments=tc["args"],
        )
        for tc in message.tool_calls
    ]


async def execute_tool_calls(
    registry: ToolRegistry,
    calls: list[NativeToolCall],
) -> list[ToolResult]:
    """Execute a batch of tool calls and return structured results.

    Tools are executed sequentially to preserve the order expected
    by the LLM conversation loop.

    Args:
        registry: The request-scoped ``ToolRegistry``.
        calls: Tool calls extracted from the LLM response.

    Returns:
        A list of ``ToolResult``, one per call, in the same order.
    """
    results: list[ToolResult] = []
    for call in calls:
        result = await registry.execute_call(call)
        results.append(result)
    return results
