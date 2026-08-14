"""ReAct agent core for the new Skill Profiling Agent (Phase 2, Phase 15).

Replaces the Phase-1 placeholder with a real native ReAct agent: the LLM
decides whether a tool is needed, receives tool results, and keeps
reasoning for multiple iterations before producing a final answer.

Loop flow:

    LLM response -> tool call -> tool execution -> tool result
        -> LLM -> more tool call(s) -> ... -> final answer

This is a generic, provider-agnostic ReAct loop. The only integration
point with existing infrastructure is ``get_skill_profiling_llm()``,
the Skill Profiling-dedicated LLM factory (TokenRouter) defined in
``skill_profiling_agent/llm.py`` — the shared ``common/llm.py`` factory
is not used by this agent. Tools are
registered per request through ``ToolRegistry``; no MCP, sandbox, or
analysis-service integration exists yet.

Since Phase 15, every successful tool result is recorded into a
request-scoped ``EvidenceBundle`` under a deterministic evidence ID, and
the final answer is deterministically validated into a structured
``SkillProfileAgentResponse.skill_profile`` (skills, proficiency,
confidence, citations, limitations). The response exposes only the
final answer, tool activity records, and the validated profile — never
chain-of-thought.

Phase 27: the run also owns one request-scoped in-memory ``ContextStore``.
Every successful tool result is stored automatically at the tool-result
boundary (tool call → tool result → Context Store → observation to the
LLM), keyed deterministically by context type (``request``,
``technologies``, ``static_analysis``, ``full_graph``,
``contributor_graph``). The LLM never decides what is stored, and the
store is created fresh per run — it is never global or shared between
requests.

Phase 28: the ``ContextStore`` prevents large Graphify outputs from
unboundedly filling the LLM context window. The original ``full_graph``
and ``contributor_graph`` outputs are always kept as-is; when one
exceeds the configurable ``graph_summary_threshold_chars``
(``AgentConfig``, safe default 4000 characters), a compact deterministic
summary (metrics and relation counts already present in the output — no
LLM involved) is stored additionally under ``full_graph_summary`` /
``contributor_graph_summary``. The original graph is never overwritten
or deleted.

Phase 29: the Profiling LLM (the existing, single LLM of the ReAct
loop) receives ONE explicit, deterministic evidence package built by
``evidence_context`` from the request-scoped ``ContextStore``. Before
every LLM invocation the package is rebuilt from the currently stored
evidence (one package at a time, the previous package message
replaced), so whichever invocation produces the final profile sees the
current stored evidence. CONTRIBUTOR mode collects ``request``,
``technologies``, ``static_analysis``, and both graphs; PROJECT mode
collects the same minus the contributor graph. Each graph is sent as
its summary when a summary exists, otherwise as the full graph — never
both — while the original graph stays stored.

Phase 30: the orchestration tool observation follows the same rule.
The raw ``analyze_contributor_repository`` envelope is never dumped
into the LLM input; the observation is rebuilt by reading the already
stored Context Store representations (``static_analysis`` + each graph
as summary-or-full, CONTRIBUTOR mode adds the contributor graph, and
PROJECT mode never does). The original envelope and full graphs remain
stored unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from sharek_agents.agents.skill_profiling.contract_schemas import SkillProfileInput
from sharek_agents.agents.skill_profiling_agent.context_store import (
    _ORCHESTRATION_TOOL,
    CONTEXT_CONTRIBUTOR_GRAPH,
    CONTEXT_CONTRIBUTOR_GRAPH_SUMMARY,
    CONTEXT_FULL_GRAPH,
    CONTEXT_FULL_GRAPH_SUMMARY,
    CONTEXT_STATIC_ANALYSIS,
    ContextStore,
)
from sharek_agents.agents.skill_profiling_agent.evidence import (
    EvidenceBundle,
    EvidenceRecord,
)
from sharek_agents.agents.skill_profiling_agent.evidence_context import (
    ANALYSIS_MODE_CONTRIBUTOR,
    analysis_mode_for_request,
    evidence_context_message,
)
from sharek_agents.agents.skill_profiling_agent.graph_summary import (
    DEFAULT_GRAPH_SUMMARY_THRESHOLD_CHARS,
)
from sharek_agents.agents.skill_profiling_agent.llm import get_skill_profiling_llm
from sharek_agents.agents.skill_profiling_agent.mcp_client import (
    MCP_PIPELINE_MAX_SECONDS,
)
from sharek_agents.agents.skill_profiling_agent.prompts import SYSTEM_PROMPT
from sharek_agents.agents.skill_profiling_agent.schemas import (
    SkillProfileAgentResponse,
    ToolActivity,
)
from sharek_agents.agents.skill_profiling_agent.skills import build_skill_profile
from sharek_agents.agents.skill_profiling_agent.tools import (
    NativeToolCall,
    Tool,
    ToolRegistry,
    ToolResult,
)

logger = logging.getLogger(__name__)

# Wall-clock budget for one full agent execution: the MCP repository-analysis
# worst case (``MCP_PIPELINE_MAX_SECONDS``) plus a margin covering the LLM
# round trips that drive the ReAct loop (bounded by ``max_iterations``,
# default 10, each call bounded by the provider per-call timeout
# ``ai_skill_profile_timeout_seconds``, default 60s).
AGENT_TIMEOUT_SECONDS: float = MCP_PIPELINE_MAX_SECONDS + 600.0


class AgentTimeoutError(Exception):
    """The agent exceeded its wall-clock execution budget."""


class AgentProviderError(Exception):
    """The LLM provider failed while the agent was executing."""


@dataclass
class AgentConfig:
    """Configuration for a single Skill Profiling Agent execution."""

    system_prompt: str = SYSTEM_PROMPT
    max_iterations: int = 10
    repetition_limit: int = 3
    timeout_seconds: float | None = AGENT_TIMEOUT_SECONDS
    llm: BaseChatModel | None = None
    tools: list[Tool] = field(default_factory=list)
    graph_summary_threshold_chars: int = DEFAULT_GRAPH_SUMMARY_THRESHOLD_CHARS


class SkillProfilingAgent:
    """Native ReAct agent with LLM tool calling.

    Each ``run()`` execution has fully isolated conversation state,
    registry, counters, repeated-call history, evidence bundle, and
    Context Store. No global mutable state is used.
    """

    def __init__(self, *, system_prompt: str = SYSTEM_PROMPT) -> None:
        self.system_prompt = system_prompt

    async def run(
        self,
        request: SkillProfileInput,
        *,
        config: AgentConfig | None = None,
    ) -> SkillProfileAgentResponse:
        """Execute the ReAct loop and return the agent response.

        The whole execution is wrapped in ``asyncio.wait_for`` so a
        wall-clock timeout cancels the in-flight LLM call.
        """
        cfg = config or AgentConfig()
        if cfg.llm is None:
            cfg.llm = get_skill_profiling_llm()

        try:
            return await asyncio.wait_for(
                self._execute(request, cfg),
                timeout=cfg.timeout_seconds,
            )
        except TimeoutError as exc:
            raise AgentTimeoutError(
                f"Skill Profiling Agent timed out after "
                f"{cfg.timeout_seconds}s",
            ) from exc

    # ── ReAct loop ----------------------------------------------------------

    async def _execute(
        self,
        request: SkillProfileInput,
        config: AgentConfig,
    ) -> SkillProfileAgentResponse:
        registry = ToolRegistry()
        for tool in config.tools:
            registry.register(tool)

        messages = self._build_initial_messages(config.system_prompt, request)
        llm = config.llm

        iteration = 0
        total_tool_calls = 0
        repeated_call_history: dict[str, int] = {}
        activities: list[ToolActivity] = []
        evidence = EvidenceBundle(request)
        context = ContextStore(
            graph_summary_threshold_chars=config.graph_summary_threshold_chars
        )
        analysis_mode = analysis_mode_for_request(request)
        evidence_context_index: int | None = None

        while iteration < config.max_iterations:
            iteration += 1

            evidence_context_index = self._append_evidence_context(
                messages,
                context,
                analysis_mode,
                evidence_context_index,
            )

            try:
                bound = self._bind_tools(llm, registry)
                response: AIMessage = await bound.ainvoke(messages)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise AgentProviderError(
                    f"LLM provider failed: {_safe_error_message(exc)}",
                ) from exc

            messages.append(response)

            calls = _extract_tool_calls(response)
            if not calls:
                return self._build_final_response(
                    request=request,
                    answer=_text(response),
                    iteration=iteration,
                    activities=activities,
                    evidence=evidence,
                )

            validated_calls: list[NativeToolCall] = []
            for call in calls:
                norm_key = _normalize_call(call)
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

            results = await self._execute_calls(registry, validated_calls)
            total_tool_calls += len(results)

            for call, result in zip(validated_calls, results):
                activities.append(_to_activity(result))
                record = evidence.record(result, call.arguments)
                if result.status == "success":
                    context.record_tool_result(result.name, result.output)
                messages.append(
                    ToolMessage(
                        content=_format_observation(
                            result,
                            record,
                            context,
                            analysis_mode,
                        ),
                        tool_call_id=result.tool_call_id,
                    ),
                )

        return self._build_max_iterations_response(
            request=request,
            iteration=iteration,
            activities=activities,
            evidence=evidence,
        )

    # ── Conversation helpers ------------------------------------------------

    @staticmethod
    def _append_evidence_context(
        messages: list,
        context: ContextStore,
        analysis_mode: str,
        previous_index: int | None,
    ) -> int | None:
        """Keep exactly one current evidence package in the LLM input.

        The package is rebuilt from the Context Store before every LLM
        invocation, so the invocation that produces the final profile
        receives one explicit, deterministic evidence package. The
        previous package message (if any) is replaced, never
        duplicated. Returns the index of the appended message, or
        ``None`` when nothing is stored for the mode.
        """
        if previous_index is not None:
            messages.pop(previous_index)
        content = evidence_context_message(context, analysis_mode)
        if content is None:
            return None
        messages.append(HumanMessage(content=content))
        return len(messages) - 1

    @staticmethod
    def _build_initial_messages(
        system_prompt: str,
        request: SkillProfileInput,
    ) -> list:
        messages: list = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        evidence_ids = ", ".join(
            repository.evidence_id for repository in request.selected_repositories
        )
        analysis_mode = analysis_mode_for_request(request)
        messages.append(
            HumanMessage(
                content=(
                    "Generate a skill profile for this request.\n\n"
                    f"Generation ID: {request.generation_id}\n"
                    f"Analysis mode: {analysis_mode}\n"
                    f"Contributor ID: {request.contributor_id}\n"
                    f"GitHub login: {request.github_login or '(none)'}\n"
                    f"Role: {request.role}\n"
                    f"Selected repositories: "
                    f"{len(request.selected_repositories)}\n"
                    f"Evidence available from the request: "
                    f"{evidence_ids}\n\n"
                    "In CONTRIBUTOR analysis the tools analyze the "
                    "contributor's code and repository ownership; in "
                    "PROJECT analysis the tools analyze each repository "
                    "as a whole and no contributor identifier exists.\n\n"
                    "Use the get_agent_context tool if you need the "
                    "repository details."
                ),
            ),
        )
        return messages

    @staticmethod
    def _bind_tools(llm: BaseChatModel, registry: ToolRegistry):
        definitions = registry.list_definitions()
        if not definitions:
            return llm

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
        return llm.bind_tools(tools)

    @staticmethod
    async def _execute_calls(
        registry: ToolRegistry,
        calls: list[NativeToolCall],
    ) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in calls:
            results.append(await registry.execute_call(call))
        return results

    # ── Response builders ---------------------------------------------------

    def _build_final_response(
        self,
        *,
        request: SkillProfileInput,
        answer: str,
        iteration: int,
        activities: list[ToolActivity],
        evidence: EvidenceBundle,
    ) -> SkillProfileAgentResponse:
        return SkillProfileAgentResponse(
            generation_id=request.generation_id,
            contributor_id=request.contributor_id,
            selected_repository_count=len(request.selected_repositories),
            iterations_used=iteration,
            tool_activities=activities,
            message=answer,
            skill_profile=build_skill_profile(
                answer,
                evidence,
                analysis_mode=analysis_mode_for_request(request),
            ),
        )

    def _build_max_iterations_response(
        self,
        *,
        request: SkillProfileInput,
        iteration: int,
        activities: list[ToolActivity],
        evidence: EvidenceBundle,
    ) -> SkillProfileAgentResponse:
        return SkillProfileAgentResponse(
            generation_id=request.generation_id,
            contributor_id=request.contributor_id,
            selected_repository_count=len(request.selected_repositories),
            iterations_used=iteration,
            tool_activities=activities,
            message=(
                "Agent reached the maximum iteration limit "
                f"({iteration}) without producing a final answer."
            ),
            skill_profile=build_skill_profile(
                "",
                evidence,
                analysis_mode=analysis_mode_for_request(request),
            ),
        )


# ── Module-level helpers ------------------------------------------------------


def _extract_tool_calls(message: AIMessage) -> list[NativeToolCall]:
    """Extract native tool calls from an LLM response.

    Empty when the LLM chose to respond with text instead.
    """
    if not hasattr(message, "tool_calls") or not message.tool_calls:
        return []

    return [
        NativeToolCall(
            id=tc["id"],
            name=tc["name"],
            arguments=tc.get("args") or {},
        )
        for tc in message.tool_calls
    ]


def _text(message: AIMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else ""


def _normalize_call(call: NativeToolCall) -> str:
    """Deterministic key for repeated-call detection.

    The key is ``tool_name|json_args`` with sorted keys for stability.
    """
    serialized = json.dumps(
        call.arguments,
        sort_keys=True,
        ensure_ascii=False,
    )
    return f"{call.name}|{serialized}"


def _format_tool_result(result: ToolResult) -> str:
    """Format a ``ToolResult`` into a string returned to the LLM."""
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


def _format_observation(
    result: ToolResult,
    record: EvidenceRecord | None,
    context: ContextStore,
    analysis_mode: str,
) -> str:
    """Format a tool observation for the LLM.

    Successful evidence-producing results carry their deterministic
    evidence ID so the LLM can cite it in the final profile. Failures
    and empty results carry no evidence ID — they are not evidence.

    The orchestration result is never exposed as its raw envelope: the
    observation is rebuilt from the already stored Context Store
    representations (``static_analysis`` plus each graph as its summary
    when one exists, otherwise as the full graph — CONTRIBUTOR mode
    adds ``contributor_graph``, PROJECT mode never does), so the LLM
    receives only one representation of each graph.
    """
    body: str
    if result.status == "success" and result.name == _ORCHESTRATION_TOOL:
        body = _build_orchestration_observation(result, context, analysis_mode)
    else:
        body = _format_tool_result(result)
    if record is None:
        return body
    return f"[evidence_id: {record.evidence_id}]\n{body}"


def _build_orchestration_observation(
    result: ToolResult,
    context: ContextStore,
    analysis_mode: str,
) -> str:
    """Rebuild the orchestration observation from the stored representations.

    Reads only the entries the Context Store already holds for the
    orchestration result — the same values the Phase-29 evidence
    package uses — so no second graph-size or summary rule exists. The
    original envelope and full graphs stay stored untouched. When the
    output could not be stored (not a parseable envelope), the raw
    output is returned as the fallback.
    """
    observation: dict[str, Any] = {}
    static_analysis = context.get(CONTEXT_STATIC_ANALYSIS)
    if static_analysis is not None:
        observation["static_analysis"] = static_analysis
    full_graph = _graph_representation(
        context, CONTEXT_FULL_GRAPH, CONTEXT_FULL_GRAPH_SUMMARY
    )
    if full_graph is not None:
        observation["full_graph"] = full_graph
    if analysis_mode == ANALYSIS_MODE_CONTRIBUTOR:
        contributor_graph = _graph_representation(
            context,
            CONTEXT_CONTRIBUTOR_GRAPH,
            CONTEXT_CONTRIBUTOR_GRAPH_SUMMARY,
        )
        if contributor_graph is not None:
            observation["contributor_graph"] = contributor_graph
    if not observation:
        return _format_tool_result(result)
    return json.dumps(observation, ensure_ascii=False)


def _graph_representation(
    context: ContextStore,
    graph_key: str,
    summary_key: str,
) -> str | None:
    """The stored summary when it exists, otherwise the stored full graph.

    Reads the existing Context Store entries; returns one
    representation only, never both. Mirrors the evidence-context
    representation rule without re-deriving graph size or summaries.
    """
    summary = context.get(summary_key)
    if summary is not None:
        return summary
    return context.get(graph_key)


def _to_activity(result: ToolResult) -> ToolActivity:
    if result.status == "success":
        return ToolActivity(
            tool=result.name,
            status="success",
            result_summary=_truncate(result.output),
        )
    return ToolActivity(
        tool=result.name,
        status="error",
        error_message=result.error_message or result.status,
    )


def _truncate(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...(truncated)"


def _safe_error_message(error: Exception) -> str:
    """Produce a safe string from an exception, stripping stack traces."""
    return f"{type(error).__name__}: {error}"
