"""Provider-selectable LLM instance management with safe caching."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter

from sharek_agents.config import settings

_cache: dict[tuple[str, str, str], BaseChatModel] = {}

_doc_understanding_cache: dict[str, ChatOpenAI] = {}


class LLMConfigurationError(ValueError):
    """Raised when the selected provider is not safely configured."""


def _required(value: str, variable_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise LLMConfigurationError(f"{variable_name} is required")
    return normalized


def _alibaba_llm(model: str) -> BaseChatModel:
    api_key = _required(settings.alibaba_api_key, "ALIBABA_API_KEY")
    base_url = _required(settings.alibaba_base_url, "ALIBABA_BASE_URL").rstrip("/")
    if not base_url.startswith("https://") or not base_url.endswith(
        "/compatible-mode/v1"
    ):
        raise LLMConfigurationError(
            "ALIBABA_BASE_URL must be an HTTPS OpenAI-compatible base URL (ending in /compatible-mode/v1)"
        )
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=settings.ai_skill_profile_timeout_seconds,
        temperature=0.0,
        extra_body={
            "enable_thinking": settings.alibaba_enable_thinking,
        },
    )


def _openrouter_llm(model: str) -> BaseChatModel:
    api_key = _required(settings.openrouter_api_key, "OPENROUTER_API_KEY")
    return ChatOpenRouter(
        model=model,
        api_key=api_key,
        timeout=int(settings.ai_skill_profile_timeout_seconds * 1000),
        temperature=0.0,
    )


def _groq_llm(model: str) -> BaseChatModel:
    api_key = _required(settings.groq_api_key, "GROQ_API_KEY")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=settings.ai_skill_profile_timeout_seconds,
        temperature=0.0,
    )


def get_llm(model: str | None = None) -> BaseChatModel:
    """Get a cached chat model for the configured provider.

    Args:
        model: Optional provider model override. The configured active model is
            used when omitted.

    Returns:
        A LangChain chat model configured for Alibaba Model Studio, OpenRouter,
        or Groq.
    """
    provider = settings.ai_provider.strip().lower()
    resolved_model = _required(
        model or settings.active_chat_model,
        {
            "alibaba": "ALIBABA_MODEL",
            "groq": "GROQ_MODEL",
        }.get(provider, "OPENROUTER_MODEL"),
    )

    if provider == "alibaba":
        base_url = settings.alibaba_base_url.strip().rstrip("/")
        cache_key = (provider, resolved_model, base_url)
        if cache_key not in _cache:
            _cache[cache_key] = _alibaba_llm(resolved_model)
        return _cache[cache_key]

    if provider == "openrouter":
        cache_key = (provider, resolved_model, "")
        if cache_key not in _cache:
            _cache[cache_key] = _openrouter_llm(resolved_model)
        return _cache[cache_key]

    if provider == "groq":
        cache_key = (provider, resolved_model, "https://api.groq.com/openai/v1")
        if cache_key not in _cache:
            _cache[cache_key] = _groq_llm(resolved_model)
        return _cache[cache_key]

    raise LLMConfigurationError(
        f"Unsupported AI_PROVIDER '{provider}'. Expected alibaba, openrouter, or groq"
    )


def get_doc_understanding_llm() -> ChatOpenAI:
    """Get a cached LLM instance for the Documentation Understanding Agent.

    Reads provider, model, base URL, and API key from settings.

    Returns:
        A ChatOpenAI instance configured for the Documentation Understanding provider.
    """
    provider = settings.doc_understanding_llm_provider
    model = settings.doc_understanding_llm_model
    base_url = settings.doc_understanding_llm_base_url
    api_key = settings.doc_understanding_llm_api_key

    # Fallback to Alibaba settings if provider is alibaba
    if provider == "alibaba":
        base_url = base_url if base_url != "https://api.moonshot.ai/v1" else settings.alibaba_base_url
        api_key = api_key or settings.alibaba_api_key
        if model == "kimi-k3" or not model:
            model = "qwen-plus"

    cache_key = f"{provider}:{model}:{base_url}"
    if cache_key not in _doc_understanding_cache:
        api_key_value = api_key if api_key else ""

        _doc_understanding_cache[cache_key] = ChatOpenAI(
            model=model,
            api_key=api_key_value,
            base_url=base_url,
            timeout=int(settings.doc_understanding_timeout_seconds * 1000),
            temperature=0.0,
        )
    return _doc_understanding_cache[cache_key]


def clear_cache() -> None:
    """Clear the LLM instance cache. Useful for testing."""
    _cache.clear()
    _doc_understanding_cache.clear()