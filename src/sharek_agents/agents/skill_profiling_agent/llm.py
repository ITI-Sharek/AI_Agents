"""Dedicated LLM factory for the Skill Profiling Agent (provider isolation).

The Skill Profiling Agent resolves its LLM ONLY through this module and
never through the shared ``common.llm.get_llm`` used by the rest of the
application. The provider is TokenRouter (OpenAI-``/v1``-compatible
endpoint); the base URL, model, and API key come from the dedicated
``SKILL_PROFILING_LLM_*`` settings, so the other features keep their
existing provider, API key, base URL, and model untouched.

The returned instance is a LangChain ``ChatOpenAI`` — the same interface
family the previous configuration (``ChatOpenRouter``) exposed — so the
existing Agent/ReAct integration keeps working unchanged:

* ``await llm.ainvoke(messages)``
* ``llm.bind_tools(...)``

The previous generation parameters (``temperature=0.0`` and the
``ai_skill_profile_timeout_seconds`` request timeout) are preserved.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from sharek_agents.config import settings

# Provider tag and safe defaults; real values are always re-read from
# settings (env-var backed) so nothing is hardcoded into the client.
_PROVIDER = "tokenrouter"

class SkillProfilingLLMConfigError(RuntimeError):
    """The dedicated Skill Profiling LLM is not configured."""


_cache: dict[str, ChatOpenAI] = {}


def get_skill_profiling_llm() -> ChatOpenAI:
    """Get the cached dedicated LLM for the Skill Profiling Agent.

    Resolves the dedicated TokenRouter configuration from the
    ``skill_profiling_llm_*`` settings:

    * provider: TokenRouter,
    * base URL: ``https://api.tokenrouter.com/v1``,
    * model: ``moonshotai/kimi-k3-free``,
    * API key: ``SKILL_PROFILING_LLM_API_KEY`` (never hardcoded, never
      logged).

    The dedicated API key is REQUIRED: without it a
    ``SkillProfilingLLMConfigError`` is raised instead of constructing a
    client, so the TokenRouter client can never silently fall back to
    any other key source (e.g. a global ``OPENAI_API_KEY`` or the
    shared common LLM key). Instances are cached per
    provider/model/base-URL combination (same caching pattern as the
    shared common LLM factory).
    """
    model = settings.skill_profiling_llm_model
    base_url = settings.skill_profiling_llm_base_url
    api_key = settings.skill_profiling_llm_api_key
    if not api_key:
        raise SkillProfilingLLMConfigError(
            "SKILL_PROFILING_LLM_API_KEY is not configured; the "
            "Skill Profiling Agent requires its dedicated TokenRouter "
            "API key"
        )

    cache_key = f"{_PROVIDER}:{model}:{base_url}"
    if cache_key not in _cache:
        _cache[cache_key] = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=int(settings.ai_skill_profile_timeout_seconds * 1000),
            temperature=0.0,
        )
    return _cache[cache_key]


def clear_cache() -> None:
    """Clear the instance cache. Useful for testing."""
    _cache.clear()


__all__ = ["SkillProfilingLLMConfigError", "clear_cache", "get_skill_profiling_llm"]