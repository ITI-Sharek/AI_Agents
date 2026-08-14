"""ReAct Agent runtime for the Documentation Understanding Agent.

A provider-agnostic, request-scoped ReAct-style Agent that uses native
LLM tool calling to iteratively gather evidence before producing a
final structured result.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from sharek_agents.agents.document_understanding.registry import ToolRegistry
from sharek_agents.agents.document_understanding.schemas import (
    DocumentUnderstandingResult,
    ValidationStatus,
)
from sharek_agents.agents.document_understanding.tool_calling import (
    bind_tools_to_llm,
    execute_tool_calls,
    extract_tool_calls,
)
from sharek_agents.agents.document_understanding.tools import (
    NativeToolCall,
    Tool,
    ToolResult,
)
from sharek_agents.common.llm import get_doc_understanding_llm


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class AgentConfig:
    """Configuration for a single ReAct Agent execution.

    All fields have defaults so callers only need to override what
    they care about.
    """

    model: str = "openai/gpt-4o"
    provider: str = "openrouter"
    system_prompt: str = ""
    tools: list[Tool] = field(default_factory=list)
    max_iterations: int = 10
    project_id: str = ""
    llm: BaseChatModel | None = None
    timeout_seconds: float | None = None
    repetition_limit: int = 3


# ── Protocol ──────────────────────────────────────────────────────────────────


class AgentRuntime(Protocol):
    """ReAct-style agent that uses native LLM tool calling.

    The runtime must:

    1. Present registered tools to the LLM as native function definitions.
    2. Let the LLM decide whether to respond with text or invoke a tool.
    3. If a tool call is returned, execute it and feed the result back.
    4. Repeat until the LLM returns a final structured answer or the
       iteration limit is reached.
    5. Return a ``DocumentUnderstandingResult`` with the final answer.
    """

    async def run(
        self,
        query: str,
        document_text: str,
        config: AgentConfig,
    ) -> DocumentUnderstandingResult:
        """Execute the ReAct loop with native tool calling.

        Args:
            query: The user's question about the document.
            document_text: Full text of the document to analyse.
            config: Agent configuration including model, provider, tools.

        Returns:
            A structured ``DocumentUnderstandingResult``.
        """
        ...


# ── Execution metadata ────────────────────────────────────────────────────────


@dataclass
class AgentResult:
    """Rich result returned by the ReAct Agent execution.

    Includes both the final structured output and execution metadata
    for observability.
    """

    result: DocumentUnderstandingResult
    iterations_used: int
    tool_calls_executed: int
    max_iterations_reached: bool
    completed_successfully: bool


# ── ReAct Agent ───────────────────────────────────────────────────────────────


class ReActAgent:
    """Request-scoped ReAct Agent with native LLM tool calling.

    Each ``run()`` execution has fully isolated:

    * conversation state
    * ToolRegistry (created from ``config.tools``)
    * execution counters (iteration count, tool-call count)
    * repeated-call history

    No global mutable state is used.
    """

    async def run(
        self,
        query: str,
        document_text: str,
        config: AgentConfig,
    ) -> DocumentUnderstandingResult:
        """Execute the ReAct loop and return the structured result.

        This is the ``AgentRuntime`` protocol-compatible entry point.
        Use ``run_with_metadata()`` when observability metadata is
        needed.
        """
        agent_result = await self._execute(query, document_text, config)
        return agent_result.result

    async def run_with_metadata(
        self,
        query: str,
        document_text: str,
        config: AgentConfig,
    ) -> AgentResult:
        """Execute the ReAct loop and return a rich result with metadata.

        This entry point returns execution observability data in
        addition to the structured result.
        """
        return await self._execute(query, document_text, config)

    # ── Internal execution ----------------------------------------------------

    async def _execute(
        self,
        query: str,
        document_text: str,
        config: AgentConfig,
    ) -> AgentResult:
        start_time = time.monotonic()

        # Build request-scoped state
        registry = ToolRegistry()
        for tool in config.tools:
            registry.register(tool)

        messages = self._build_initial_messages(
            config.system_prompt,
            query,
            document_text,
        )

        llm = config.llm or get_doc_understanding_llm()

        iteration = 0
        total_tool_calls = 0
        repeated_call_history: dict[str, int] = {}

        while iteration < config.max_iterations:
            # Check timeout at start of each iteration
            if config.timeout_seconds is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= config.timeout_seconds:
                    return self._timeout_result(config, iteration, total_tool_calls)

            iteration += 1

            # Invoke the LLM with native tool definitions bound
            try:
                bound = bind_tools_to_llm(llm, registry)
                response: AIMessage = await bound.ainvoke(messages)
            except Exception as exc:
                return self._provider_error_result(
                    config, iteration, total_tool_calls, exc,
                )

            messages.append(response)

            # Extract native tool calls from the LLM response
            calls = extract_tool_calls(response)

            if not calls:
                # LLM responded with text — final answer
                return self._finalize(
                    response=response,
                    config=config,
                    iteration=iteration,
                    total_tool_calls=total_tool_calls,
                )

            # Process tool calls: validate repetitions, execute
            validated_calls: list[NativeToolCall] = []
            for call in calls:
                norm_key = self._normalize_call(call)
                count = repeated_call_history.get(norm_key, 0) + 1
                repeated_call_history[norm_key] = count

                if count > config.repetition_limit:
                    msg = (
                        f"Tool '{call.name}' with arguments "
                        f"{json.dumps(call.arguments)} has been called "
                        f"{count} times. Try a different approach or "
                        f"proceed with the information you already have."
                    )
                    messages.append(ToolMessage(content=msg, tool_call_id=call.id))
                    continue

                validated_calls.append(call)

            if not validated_calls:
                continue

            # Execute tools sequentially through ToolRegistry
            results = await execute_tool_calls(registry, validated_calls)
            total_tool_calls += len(results)

            # Append structured tool results to conversation
            for result in results:
                content = self._format_tool_result(result)
                messages.append(
                    ToolMessage(content=content, tool_call_id=result.tool_call_id),
                )

        # Max iterations reached without a final answer
        return self._max_iterations_result(config, iteration, total_tool_calls)

    # ── Conversation helpers --------------------------------------------------

    @staticmethod
    def _build_initial_messages(
        system_prompt: str,
        query: str,
        document_text: str,
    ) -> list:
        messages: list = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        user_content = f"Query: {query}\n\nDocument:\n{document_text}"
        messages.append(HumanMessage(content=user_content))
        return messages

    @staticmethod
    def _normalize_call(call: NativeToolCall) -> str:
        """Deterministic key for repeated-call detection.

        The key is ``tool_name|json_args`` where arguments are
        serialised with sorted keys for stability.
        """
        serialized = json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
        return f"{call.name}|{serialized}"

    @staticmethod
    def _format_tool_result(result: ToolResult) -> str:
        """Format a ``ToolResult`` into a human-readable string for the LLM."""
        if result.status == "success":
            return result.output
        if result.status == "empty":
            return f"(empty result from tool '{result.name}')"
        if result.status == "not_found":
            return f"Tool '{result.name}' is not available."
        if result.status == "validation_error":
            return (
                f"Validation error for tool '{result.name}': "
                f"{result.error_message}"
            )
        if result.status == "execution_error":
            return (
                f"Execution error for tool '{result.name}': "
                f"{result.error_message}"
            )
        return result.error_message or "(unknown tool result)"

    # ── Final output handling -------------------------------------------------

    def _finalize(
        self,
        response: AIMessage,
        config: AgentConfig,
        iteration: int,
        total_tool_calls: int,
    ) -> AgentResult:
        """Validate the LLM's final response and produce the ``AgentResult``."""
        content = response.content or ""
        parsed = self._parse_structured_output(content, config.project_id)

        return AgentResult(
            result=parsed,
            iterations_used=iteration,
            tool_calls_executed=total_tool_calls,
            max_iterations_reached=False,
            completed_successfully=parsed.validation_status.is_valid,
        )

    @staticmethod
    def _parse_structured_output(
        content: str,
        project_id: str,
    ) -> DocumentUnderstandingResult:
        """Parse and validate the LLM's final text as a ``DocumentUnderstandingResult``.

        Tries direct JSON parse, then markdown code-block extraction.
        Returns a result with ``validation_status.is_valid=False`` when
        parsing fails.
        """
        fallback_pid = project_id or "unknown"

        if not content:
            return DocumentUnderstandingResult(
                project_id=fallback_pid,
                validation_status=ValidationStatus(
                    is_valid=False,
                    warnings=["LLM returned empty response"],
                ),
            )

        # Attempt 1: direct JSON parse into the schema
        try:
            result = DocumentUnderstandingResult.model_validate_json(content)
            if project_id and not result.project_id:
                result.project_id = project_id
            result.validation_status = ValidationStatus(
                is_valid=True,
                missing_required=[],
                warnings=[],
            )
            return result
        except Exception:
            pass

        # Attempt 2: extract JSON from markdown code block(s)
        extracted = _extract_json(text=content)
        if extracted is not None:
            try:
                result = DocumentUnderstandingResult.model_validate(extracted)
                if project_id and not result.project_id:
                    result.project_id = project_id
                result.validation_status = ValidationStatus(
                    is_valid=True,
                    missing_required=[],
                    warnings=[],
                )
                return result
            except Exception:
                pass

        # Could not parse — return an invalid result
        return DocumentUnderstandingResult(
            project_id=fallback_pid,
            validation_status=ValidationStatus(
                is_valid=False,
                warnings=[
                    "Failed to parse LLM response as "
                    "DocumentUnderstandingResult.",
                ],
            ),
        )

    # ── Error / edge-case result builders ------------------------------------

    @staticmethod
    def _timeout_result(
        config: AgentConfig,
        iteration: int,
        total_tool_calls: int,
    ) -> AgentResult:
        return AgentResult(
            result=DocumentUnderstandingResult(
                project_id=config.project_id or "unknown",
                validation_status=ValidationStatus(
                    is_valid=False,
                    warnings=[
                        f"Agent execution timed out after "
                        f"{config.timeout_seconds}s",
                    ],
                ),
            ),
            iterations_used=iteration,
            tool_calls_executed=total_tool_calls,
            max_iterations_reached=False,
            completed_successfully=False,
        )

    @staticmethod
    def _provider_error_result(
        config: AgentConfig,
        iteration: int,
        total_tool_calls: int,
        error: Exception,
    ) -> AgentResult:
        safe_msg = _safe_error_message(error)
        return AgentResult(
            result=DocumentUnderstandingResult(
                project_id=config.project_id or "unknown",
                validation_status=ValidationStatus(
                    is_valid=False,
                    warnings=[f"LLM provider error: {safe_msg}"],
                ),
            ),
            iterations_used=iteration,
            tool_calls_executed=total_tool_calls,
            max_iterations_reached=False,
            completed_successfully=False,
        )

    @staticmethod
    def _max_iterations_result(
        config: AgentConfig,
        iteration: int,
        total_tool_calls: int,
    ) -> AgentResult:
        return AgentResult(
            result=DocumentUnderstandingResult(
                project_id=config.project_id or "unknown",
                validation_status=ValidationStatus(
                    is_valid=False,
                    warnings=[
                        f"Agent reached maximum iteration limit "
                        f"({config.max_iterations})",
                    ],
                ),
            ),
            iterations_used=iteration,
            tool_calls_executed=total_tool_calls,
            max_iterations_reached=True,
            completed_successfully=False,
        )


# ── Module-level helpers ──────────────────────────────────────────────────────


def _extract_json(text: str) -> dict[str, Any] | None:
    """Try to extract a JSON object from text, including markdown blocks."""
    # Look for ```json ... ``` blocks
    pattern = r"```(?:json)?\s*\n?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try treating the entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _safe_error_message(error: Exception) -> str:
    """Produce a safe string from an exception, stripping stack traces."""
    return f"{type(error).__name__}: {error}"
