import json
import os
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

from sharek_agents.agents.skill_profiling.prompts import SYSTEM_PROMPT
from sharek_agents.agents.skill_profiling.schemas import AgentResponse, SkillProfilingResult
from sharek_agents.agents.skill_profiling.tools import gather_all_evidence

DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "openai/gpt-oss-120b"


try:
    from anthropic import (
        APIError as AnthropicAPIError,
        AuthenticationError as AnthropicAuthError,
        RateLimitError as AnthropicRateLimitError,
    )
except ImportError:
    AnthropicAPIError = type("_AnthropicAPIError", (Exception,), {})
    AnthropicAuthError = type("_AnthropicAuthError", (Exception,), {})
    AnthropicRateLimitError = type("_AnthropicRateLimitError", (Exception,), {})

try:
    from openai import (
        APIError as OpenAIAPIError,
        AuthenticationError as OpenAIAuthError,
        RateLimitError as OpenAIRateLimitError,
    )
except ImportError:
    OpenAIAPIError = type("_OpenAIAPIError", (Exception,), {})
    OpenAIAuthError = type("_OpenAIAuthError", (Exception,), {})
    OpenAIRateLimitError = type("_OpenAIRateLimitError", (Exception,), {})


@lru_cache(maxsize=1)
def get_llm():
    return init_chat_model(
        os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        model_provider=os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER),
    )


async def _invoke_llm(
    prompt: ChatPromptTemplate, structured: object, evidence_json: str
) -> SkillProfilingResult:
    if os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).lower() == "groq":
        return await _invoke_groq_json(evidence_json)

    try:
        return await (prompt | structured).ainvoke({"evidence": evidence_json})
    except (ValidationError, ValueError, TypeError):
        retry_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", (
                "Your previous response did not match the required schema. "
                "Fix the format.\n\nEvidence:\n\n{evidence}"
            )),
        ])
        return await (retry_prompt | structured).ainvoke({"evidence": evidence_json})


async def _invoke_groq_json(evidence_json: str) -> SkillProfilingResult:
    if ChatGroq is None:
        raise RuntimeError("langchain-groq is not installed")

    llm = ChatGroq(
        model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        temperature=0,
        timeout=float(os.environ.get("LLM_TIMEOUT_SECONDS", "45")),
        max_retries=1,
    )
    messages = [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Return ONLY valid JSON matching this schema: "
            "{\"skills\":[{\"name\": string, \"evidence_type\": \"github_stats|static_analysis|graphify_relations\", \"description\": string, \"supporting_evidence\": [string], \"evidence_ids\": [exact evidence_id strings]}], \"overall_level\": string, \"summary\": string}. "
            "Do not wrap in markdown. Evidence:\n\n" + evidence_json,
        ),
    ]
    response = await llm.ainvoke(messages)
    content = response.content if isinstance(response.content, str) else str(response.content)
    payload = _extract_json_object(content)
    return SkillProfilingResult.model_validate(payload)


def _extract_json_object(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Groq response did not contain a JSON object")
    return json.loads(cleaned[start : end + 1])


async def run(username: str, evidence: dict | None = None) -> AgentResponse:
    if evidence is None:
        evidence = await gather_all_evidence(username)
    evidence_json = json.dumps(evidence, indent=2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Profile the following developer based on this evidence:\n\n{evidence}"),
    ])

    try:
        structured = (
            None
            if os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER).lower() == "groq"
            else get_llm().with_structured_output(SkillProfilingResult)
        )
        result = await _invoke_llm(prompt, structured, evidence_json)
        return AgentResponse(status="success", data=result)
    except (OpenAIAuthError, AnthropicAuthError):
        return AgentResponse(
            status="failed",
            error_code="llm_provider_error",
            retryable=False,
        )
    except (OpenAIRateLimitError, AnthropicRateLimitError, OpenAIAPIError, AnthropicAPIError):
        return AgentResponse(
            status="failed",
            error_code="llm_provider_error",
            retryable=True,
        )
    except Exception:
        return AgentResponse(
            status="failed",
            error_code="llm_provider_error",
            retryable=True,
        )
