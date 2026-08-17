"""Custom ReAct-style Gap Guidance Agent (Phase 2) with explicit agentic
RAG retrieval (Phase 3) and a combined final response (Phase 4).

A small, self-contained agent. The LLM receives the complete
``AdvisoryFitResult`` and drives an explicit retrieval loop:

    analyze gaps (structured)
        -> determine required roadmap knowledge
        -> call search_roadmap
        -> observe retrieved material
        -> evaluate retrieval sufficiency (structured verdict)
        -> if insufficient: identify missing knowledge, refine the query,
           and search again (bounded retrieval rounds)
        -> if sufficient or the retrieval limit is reached:
           generate the final guidance (learningGuidance + practiceRoadmap)

Python owns the loop control, the structured boundaries (gap analysis and
sufficiency verdicts are validated), the tool contract enforcement, the
bounded limits, and the final structured-output validation. The LLM owns
gap identification, prioritization, knowledge-need formulation, query
refinement, sufficiency judgement, and the generation of guidance and
roadmap.

The final response (Phase 4) is ONE combined result: the exact
``AdvisoryFitResult`` received by the endpoint (attached by Python from
the run input — the LLM never echoes or recalculates it), ONE combined
``learningGuidance`` covering all relevant gaps, and ONE combined
``practiceRoadmap`` string. The LLM's final output carries only the two
generated fields; Python validates them strictly and attaches the
authoritative Advisory Fit result.

The loop is bounded by ``max_iterations`` (reasoning turns),
``max_tool_calls`` (total tool executions), and ``max_retrieval_rounds``
(retrieval rounds, each round = one reasoning turn with search calls plus
one sufficiency verdict). The agent can never loop indefinitely.

No agent framework is used: the loop is implemented directly on top of the
dedicated Gap Guidance LLM factory (``gap_guidance/llm.py``
``get_gap_guidance_llm``) and the LangChain message types, exactly like
the rest of the repository's agents. All state is request-scoped; there is
no memory. After ``run()`` the completed run state is available on
``last_state`` for observability.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field, ValidationError

from sharek_agents.agents.advisory_fit.prompts import output_language_instruction
from sharek_agents.agents.advisory_fit.schemas import (
    AdvisoryFitResult,
    ContractModel,
)
from sharek_agents.agents.gap_guidance.llm import get_gap_guidance_llm
from sharek_agents.agents.gap_guidance.prompts import SYSTEM_PROMPT
from sharek_agents.agents.gap_guidance.schemas import GapGuidanceResult
from sharek_agents.agents.gap_guidance.tools import (
    NativeToolCall,
    SearchRoadmapTool,
    Tool,
    ToolRegistry,
    ToolResult,
)

logger = logging.getLogger(__name__)

SEARCH_ROADMAP_TOOL = "search_roadmap"

DEFAULT_MAX_ITERATIONS = 6
DEFAULT_MAX_TOOL_CALLS = 6
DEFAULT_MAX_RETRIEVAL_ROUNDS = 3
DEFAULT_TIMEOUT_SECONDS = 120.0

RETRIEVAL_LIMIT_MESSAGE = (
    "The retrieval limit has been reached and the retrieved material is "
    "still insufficient. Produce the final answer now. Clearly state in "
    "learningGuidance that sufficient roadmap material was unavailable, "
    "and keep the practice roadmap grounded in what was actually retrieved "
    "— do not invent roadmap content."
)


class GapGuidanceProviderError(Exception):
    """The Gap Guidance provider (LLM or agent) produced an invalid outcome."""


class GapGuidanceProviderTimeout(GapGuidanceProviderError):
    """The Gap Guidance agent exceeded its wall-clock execution budget."""


class GapItem(BaseModel):
    """One gap identified by the LLM from the Advisory Fit result."""

    skill: str = Field(min_length=1, max_length=200)
    current_level: str | None = Field(default=None, max_length=50)
    target_level: str | None = Field(default=None, max_length=50)
    gap_type: Literal["LOWER", "MISSING", "NOT_EVIDENCED"]


class GapAnalysis(BaseModel):
    """Structured output of the first reasoning stage.

    The LLM identifies the meaningful gaps; the Agent only records them.
    """

    gaps: list[GapItem] = Field(default_factory=list)


class RetrievalSufficiencyVerdict(BaseModel):
    """Structured evaluation of one retrieval round.

    ``sufficient`` when the retrieved material is enough to produce a
    grounded final answer; otherwise ``missing_knowledge`` names what is
    still needed so the next search can be refined.
    """

    sufficient: bool
    missing_knowledge: list[str] = Field(default_factory=list)
    reason: str = Field(default="", max_length=1000)


class FinalGuidance(ContractModel):
    """Strict contract for the LLM's final synthesis output (Phase 4).

    The LLM generates ONLY the two combined fields: one ``learningGuidance``
    covering all relevant gaps and one ``practiceRoadmap`` string ordering
    them into a single coherent path. The authoritative
    ``advisoryFitResult`` is NOT part of this output: Python attaches it
    unchanged from the run input when building ``GapGuidanceResult``, so the
    API response always carries the exact result the endpoint received.
    Extra keys (e.g. an echoed advisoryFitResult) are rejected.
    """

    learning_guidance: str
    practice_roadmap: str


@dataclass
class GapGuidanceAgentConfig:
    """Configuration for a single Gap Guidance Agent execution."""

    system_prompt: str = SYSTEM_PROMPT
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_retrieval_rounds: int = DEFAULT_MAX_RETRIEVAL_ROUNDS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    llm: BaseChatModel | None = None
    tools: list[Tool] = field(default_factory=lambda: [SearchRoadmapTool()])


@dataclass
class RetrievalRound:
    """One retrieval round: search requests, retrieved chunks, and verdict."""

    requests: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    verdict: dict[str, Any] | None = None


@dataclass
class GapGuidanceRunState:
    """Request-scoped state tracked during one Agent execution.

    Makes the retrieval process observable: the original result, the gaps
    identified by the LLM, every retrieval round (requests, chunks, and the
    sufficiency verdict), the knowledge still considered missing, and the
    final validated result.
    """

    advisory_fit_result: AdvisoryFitResult
    identified_gaps: list[dict[str, Any]] = field(default_factory=list)
    retrieval_rounds: list[RetrievalRound] = field(default_factory=list)
    missing_knowledge: list[str] = field(default_factory=list)
    final_result: GapGuidanceResult | None = None


class GapGuidanceAgent:
    """Small ReAct-style agent producing a ``GapGuidanceResult``.

    Each ``run()`` has fully isolated conversation state, tool registry,
    and run state. No global mutable state is used.
    """

    def __init__(self, *, config: GapGuidanceAgentConfig | None = None) -> None:
        self._config = config or GapGuidanceAgentConfig()
        self._last_state: GapGuidanceRunState | None = None

    @property
    def last_state(self) -> GapGuidanceRunState | None:
        """The completed run state of the most recent ``run()``.

        Request-scoped observability only: refreshed on every run and never
        persisted.
        """
        return self._last_state

    async def run(
        self,
        advisory_fit_result: AdvisoryFitResult,
        *,
        answer: str = "",
    ) -> GapGuidanceResult:
        """Execute the agentic retrieval loop and return the validated result.

        ``answer`` is the request-selected natural-language output language
        (free-form language name; empty means the default English behavior).
        It is injected into the system prompt as a high-priority instruction
        and only controls the language of the generated guidance/roadmap
        text; the preserved ``AdvisoryFitResult`` values are never affected.

        The whole execution is wrapped in ``asyncio.wait_for`` so a
        wall-clock timeout cancels the in-flight LLM call.
        """
        cfg = self._config
        if cfg.llm is None:
            cfg.llm = get_gap_guidance_llm()

        try:
            return await asyncio.wait_for(
                self._execute(advisory_fit_result, cfg, answer),
                timeout=cfg.timeout_seconds,
            )
        except TimeoutError as exc:
            raise GapGuidanceProviderTimeout(
                f"Gap Guidance agent timed out after {cfg.timeout_seconds}s",
            ) from exc

    # ── Agentic retrieval loop ----------------------------------------------

    async def _execute(
        self,
        advisory_fit_result: AdvisoryFitResult,
        config: GapGuidanceAgentConfig,
        answer: str = "",
    ) -> GapGuidanceResult:
        registry = ToolRegistry()
        for tool in config.tools:
            registry.register(tool)

        messages = self._build_initial_messages(
            config.system_prompt, advisory_fit_result, answer
        )
        llm = config.llm
        state = GapGuidanceRunState(advisory_fit_result=advisory_fit_result)
        total_tool_calls = 0
        reasoning_turns = 0

        analysis = await self._invoke_structured(llm, GapAnalysis, messages)
        state.identified_gaps = [gap.model_dump() for gap in analysis.gaps]
        messages.append(self._analysis_message(analysis))

        verdict = RetrievalSufficiencyVerdict(sufficient=False)

        for _ in range(config.max_retrieval_rounds):
            reasoning_turns += 1
            if reasoning_turns > config.max_iterations:
                raise GapGuidanceProviderError(
                    "Gap Guidance agent reached the maximum iteration limit "
                    f"({config.max_iterations}) without producing a final answer",
                )

            response = await self._invoke_llm(llm, registry, messages)
            messages.append(response)

            calls = _extract_tool_calls(response)
            if not calls:
                return self._finalize(response, state)

            round_state = RetrievalRound()
            for call in calls:
                if total_tool_calls >= config.max_tool_calls:
                    raise GapGuidanceProviderError(
                        "Gap Guidance agent reached the maximum tool call "
                        f"limit ({config.max_tool_calls}) without producing "
                        "a final answer",
                    )
                result = await registry.execute_call(call)
                total_tool_calls += 1
                self._record_round(round_state, call, result)
                messages.append(
                    ToolMessage(
                        content=_format_tool_result(result),
                        tool_call_id=result.tool_call_id,
                    ),
                )

            verdict = await self._invoke_structured(
                llm, RetrievalSufficiencyVerdict, messages
            )
            round_state.verdict = verdict.model_dump()
            state.retrieval_rounds.append(round_state)
            for knowledge in verdict.missing_knowledge:
                if knowledge not in state.missing_knowledge:
                    state.missing_knowledge.append(knowledge)

            if verdict.sufficient:
                break
            messages.append(self._insufficiency_message(verdict))

        reasoning_turns += 1
        if reasoning_turns > config.max_iterations:
            raise GapGuidanceProviderError(
                "Gap Guidance agent reached the maximum iteration limit "
                f"({config.max_iterations}) without producing a final answer",
            )
        if not verdict.sufficient:
            messages.append(HumanMessage(content=RETRIEVAL_LIMIT_MESSAGE))

        response = await self._invoke_llm(llm, registry, messages)
        messages.append(response)
        if _extract_tool_calls(response):
            raise GapGuidanceProviderError(
                "Gap Guidance agent reached the maximum retrieval limit "
                f"({config.max_retrieval_rounds} rounds) without producing "
                "a final answer",
            )

        return self._finalize(response, state)

    # ── Conversation helpers ------------------------------------------------

    @staticmethod
    def _build_initial_messages(
        system_prompt: str,
        advisory_fit_result: AdvisoryFitResult,
        answer: str = "",
    ) -> list:
        messages: list = []
        if system_prompt:
            content = system_prompt
            language_instruction = output_language_instruction(answer)
            if language_instruction:
                content = f"{content}\n\n{language_instruction}"
            messages.append(SystemMessage(content=content))
        elif answer.strip():
            messages.append(
                SystemMessage(content=output_language_instruction(answer))
            )
        messages.append(
            HumanMessage(
                content=(
                    "Generate ONE combined learning guidance and ONE "
                    "combined ordered practice roadmap (a single string) "
                    "from this Advisory Fit result (source of truth):\n\n"
                    + json.dumps(
                        advisory_fit_result.model_dump(mode="json", by_alias=True),
                        indent=2,
                        ensure_ascii=False,
                    )
                ),
            ),
        )
        return messages

    @staticmethod
    def _analysis_message(analysis: GapAnalysis) -> HumanMessage:
        return HumanMessage(
            content=(
                "Gap analysis (authoritative for this run):\n"
                + json.dumps(analysis.model_dump(), indent=2, ensure_ascii=False)
            ),
        )

    @staticmethod
    def _insufficiency_message(
        verdict: RetrievalSufficiencyVerdict,
    ) -> HumanMessage:
        missing = verdict.missing_knowledge or ["(not specified)"]
        return HumanMessage(
            content=(
                "Retrieval sufficiency evaluation: INSUFFICIENT.\n"
                f"Reason: {verdict.reason or 'the retrieved material is not sufficient'}\n"
                f"Missing knowledge: {json.dumps(missing, ensure_ascii=False)}\n"
                "If additional retrieval can help, call search_roadmap again "
                "with a more focused query addressing the missing knowledge. "
                "Otherwise produce the final answer with the material you have."
            ),
        )

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
    async def _invoke_llm(
        llm: BaseChatModel,
        registry: ToolRegistry,
        messages: list,
    ) -> AIMessage:
        try:
            bound = GapGuidanceAgent._bind_tools(llm, registry)
            return await bound.ainvoke(messages)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise GapGuidanceProviderError(
                f"LLM provider failed: {_safe_error_message(exc)}",
            ) from exc

    @staticmethod
    async def _invoke_structured(
        llm: BaseChatModel,
        schema: type[BaseModel],
        messages: list,
    ) -> BaseModel:
        try:
            structured = llm.with_structured_output(schema)
            result = await structured.ainvoke(messages)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise GapGuidanceProviderError(
                f"LLM provider failed: {_safe_error_message(exc)}",
            ) from exc
        if not isinstance(result, schema):
            raise GapGuidanceProviderError(
                "Gap Guidance agent returned invalid structured output",
            )
        return result

    @staticmethod
    def _record_round(
        round_state: RetrievalRound,
        call: NativeToolCall,
        result: ToolResult,
    ) -> None:
        """Record retrieval requests and retrieved chunks into the round state."""
        if call.name != SEARCH_ROADMAP_TOOL:
            return
        round_state.requests.append(call.arguments)
        if result.status != "success":
            return
        try:
            parsed = json.loads(result.output)
        except json.JSONDecodeError:
            return
        if isinstance(parsed, dict):
            results = parsed.get("results")
            if isinstance(results, list):
                round_state.chunks.extend(
                    item for item in results if isinstance(item, dict)
                )

    def _finalize(
        self,
        response: AIMessage,
        state: GapGuidanceRunState,
    ) -> GapGuidanceResult:
        """Validate the LLM's final JSON answer and build the combined result.

        The LLM output must strictly match ``FinalGuidance`` (the two
        generated fields only). The authoritative ``advisoryFitResult`` is
        attached here from the run input, never from the LLM output, so the
        API response preserves the exact Advisory Fit result received by
        the endpoint.
        """
        text = _text(response)
        payload = _strip_code_fence(text)
        try:
            guidance = FinalGuidance.model_validate_json(payload)
        except ValidationError as exc:
            raise GapGuidanceProviderError(
                "Gap Guidance agent returned invalid output; expected a "
                "FinalGuidance JSON object",
            ) from exc
        final = GapGuidanceResult(
            advisory_fit_result=state.advisory_fit_result,
            learning_guidance=guidance.learning_guidance,
            practice_roadmap=guidance.practice_roadmap,
        )
        state.final_result = final
        self._last_state = state
        return final


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


def _strip_code_fence(text: str) -> str:
    """Strip a ```json ... ``` or ``` ... ``` fence around the answer."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


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


def _safe_error_message(error: Exception) -> str:
    """Produce a safe string from an exception, stripping stack traces."""
    return f"{type(error).__name__}: {error}"