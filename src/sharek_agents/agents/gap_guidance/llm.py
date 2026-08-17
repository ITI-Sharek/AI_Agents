"""Dedicated LLM factory for the Gap Guidance Agent (configuration isolation).

The Gap Guidance Agent resolves its LLM ONLY through this module and never
through the shared ``common.llm.get_llm`` used by the rest of the
application. The provider, model, API key, and base URL come from the
dedicated ``GAP_GUIDANCE_LLM_*`` settings, so the other features keep their
existing provider, API key, base URL, and model untouched.

Dedicated mode is active when both ``GAP_GUIDANCE_LLM_PROVIDER`` and
``GAP_GUIDANCE_LLM_MODEL`` are set. For the ``openrouter`` provider a
``ChatOpenRouter`` is built (same client family as the shared factory);
for any other provider an OpenAI-``/v1``-compatible ``ChatOpenAI`` is
built with the dedicated base URL. The dedicated API key
(``GAP_GUIDANCE_LLM_API_KEY``) is passed through as configured — never
hardcoded, never logged — and the previous generation parameters
(``temperature=0.0`` and the ``ai_skill_profile_timeout_seconds`` request
timeout) are preserved.

When the dedicated provider or model is missing, the factory falls back to
the existing shared OpenRouter configuration
(``common.llm.get_llm``), preserving the previous runtime behavior
exactly.

The returned instance is a LangChain chat model — the same interface
family the previous configuration (``ChatOpenRouter``) exposed — so the
existing agent loop keeps working unchanged:

* ``await llm.ainvoke(messages)``
* ``llm.bind_tools(...)``
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter

from sharek_agents.common.llm import get_llm
from sharek_agents.config import settings

_CACHE: dict[str, BaseChatModel] = {}


def get_gap_guidance_llm() -> BaseChatModel:
    """Get the dedicated LLM for the Gap Guidance Agent.

    Dedicated mode is active when both ``GAP_GUIDANCE_LLM_PROVIDER`` and
    ``GAP_GUIDANCE_LLM_MODEL`` are set; the dedicated API key and base URL
    are passed through as configured. Instances are cached per
    provider/model/base-URL combination (same caching pattern as the
    shared common LLM factory). When the dedicated provider or model is
    missing, the existing shared configuration is used instead.
    """
    provider = settings.gap_guidance_llm_provider
    model = settings.gap_guidance_llm_model
    if not provider or not model:
        return get_llm()

    api_key = settings.gap_guidance_llm_api_key
    base_url = settings.gap_guidance_llm_base_url

    cache_key = f"{provider}:{model}:{base_url}"
    if cache_key not in _CACHE:
        timeout_ms = int(settings.ai_skill_profile_timeout_seconds * 1000)
        if provider == "openrouter":
            _CACHE[cache_key] = ChatOpenRouter(
                model=model,
                api_key=api_key,
                timeout=timeout_ms,
                temperature=0.0,
            )
        else:
            _CACHE[cache_key] = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url or None,
                timeout=timeout_ms,
                temperature=0.0,
            )
    return _CACHE[cache_key]


def clear_cache() -> None:
    """Clear the instance cache. Useful for testing."""
    _CACHE.clear()


__all__ = ["clear_cache", "get_gap_guidance_llm"]
