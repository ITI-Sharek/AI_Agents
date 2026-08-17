"""Tool abstractions and the ``search_roadmap`` tool for the Gap Guidance Agent.

The tool contract mirrors the repository's existing native-tool-calling
pattern (``Tool`` protocol, ``ToolRegistry``, ``NativeToolCall``,
``ToolResult``), kept local to the Gap Guidance module so the module stays
self-contained.

``search_roadmap`` is the ONLY tool of this phase. It is a pure retrieval
interface: it decides nothing about gaps, fit, scores, guidance, or the
final roadmap. It forwards a focused retrieval request to a
``RoadmapRetriever`` (the Phase 2 mock backend by default; a real RAG
backend later implements the same protocol without any change to the tool
contract).
"""

from __future__ import annotations

import json
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator

from sharek_agents.agents.gap_guidance.retrieval import (
    MockRoadmapRetriever,
    RoadmapRetriever,
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

    The ``status`` field distinguishes execution outcomes so the Agent can
    feed failures back to the LLM without inspecting exception types.
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


class ToolRegistry:
    """Request-scoped registry of tools available to one Agent execution.

    Never shared between requests; provides deterministic (insertion-ordered)
    tool listing.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolRegistryError(f"Tool '{tool.name}' is already registered")
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
        ``ToolResult`` objects (never propagated) so the Agent can feed the
        failure back to the LLM safely.
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


class SearchRoadmapInput(BaseModel):
    """Arguments accepted by the ``search_roadmap`` tool.

    ``skill`` and ``query`` are required; the levels and gap description are
    optional context the Agent supplies to help the retrieval layer find
    relevant roadmap material.
    """

    skill: str = Field(min_length=1, max_length=200, description="The skill to search roadmap knowledge for")
    query: str = Field(min_length=1, max_length=500, description="Focused search query describing the roadmap knowledge needed")
    current_level: str | None = Field(default=None, max_length=50, description="The contributor's current level for the skill")
    target_level: str | None = Field(default=None, max_length=50, description="The required/target level for the skill")
    gap_description: str | None = Field(default=None, max_length=1000, description="Short description of the skill gap")
    limit: int = Field(default=3, ge=1, le=10, description="Maximum number of roadmap chunks to return")

    @field_validator("skill", "query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("skill and query must not be empty")
        return stripped


class SearchRoadmapTool:
    """Retrieval-only tool: find relevant roadmap knowledge chunks.

    The tool performs NO analysis: it does not decide whether a contributor
    has a gap, does not calculate fit, does not score, and does not generate
    guidance or roadmaps. It only forwards the retrieval request to the
    configured ``RoadmapRetriever`` and serializes the returned chunks for
    the LLM.
    """

    def __init__(self, retriever: RoadmapRetriever | None = None) -> None:
        self._retriever: RoadmapRetriever = retriever or MockRoadmapRetriever()

    @property
    def name(self) -> str:
        return "search_roadmap"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Retrieve roadmap knowledge chunks for a skill gap. Call this "
                "when you need practice/learning material to build the final "
                "roadmap. Returns relevant roadmap chunks (skill, topic, "
                "content) matching the skill and query; may return no chunks "
                "when nothing matches. This tool only retrieves — it does not "
                "decide whether a gap exists."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "skill": {
                        "type": "string",
                        "description": "The skill to search roadmap knowledge for",
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Focused search query describing the roadmap "
                            "knowledge needed (e.g. 'clean architecture', "
                            "'query optimization')"
                        ),
                    },
                    "current_level": {
                        "type": ["string", "null"],
                        "description": "The contributor's current level for the skill",
                    },
                    "target_level": {
                        "type": ["string", "null"],
                        "description": "The required/target level for the skill",
                    },
                    "gap_description": {
                        "type": ["string", "null"],
                        "description": "Short description of the skill gap",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of roadmap chunks to return",
                        "default": 3,
                    },
                },
                "required": ["skill", "query"],
                "additionalProperties": False,
            },
        )

    async def execute(self, **kwargs: Any) -> str:
        args = SearchRoadmapInput.model_validate(kwargs)
        chunks = await self._retriever.search(
            skill=args.skill,
            query=args.query,
            current_level=args.current_level,
            target_level=args.target_level,
            gap_description=args.gap_description,
            limit=args.limit,
        )
        return json.dumps(
            {
                "query": args.query,
                "count": len(chunks),
                "results": [chunk.model_dump(mode="json") for chunk in chunks],
            },
            ensure_ascii=False,
        )