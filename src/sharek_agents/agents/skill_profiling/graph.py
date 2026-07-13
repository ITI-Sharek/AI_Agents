import json
import os
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from sharek_agents.agents.skill_profiling.prompts import SYSTEM_PROMPT
from sharek_agents.agents.skill_profiling.schemas import AgentResponse, SkillProfilingResult
from sharek_agents.agents.skill_profiling.tools import gather_all_evidence


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
        os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        model_provider=os.environ.get("LLM_PROVIDER", "openai"),
    )


async def _invoke_llm(
    prompt: ChatPromptTemplate, structured: object, evidence_json: str
) -> SkillProfilingResult:
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


async def run(username: str, evidence: dict | None = None) -> AgentResponse:
    if evidence is None:
        evidence = await gather_all_evidence(username)
    evidence_json = json.dumps(evidence, indent=2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Profile the following developer based on this evidence:\n\n{evidence}"),
    ])

    structured = get_llm().with_structured_output(SkillProfilingResult)

    try:
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
