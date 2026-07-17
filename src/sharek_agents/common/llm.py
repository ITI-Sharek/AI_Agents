"""LLM instance management with multi-model caching."""

from langchain_openrouter import ChatOpenRouter

from sharek_agents.config import settings

_cache: dict[str, ChatOpenRouter] = {}


def get_llm(model: str | None = None) -> ChatOpenRouter:
    """Get a cached LLM instance for the given model.

    Args:
        model: Model identifier on OpenRouter. If None, uses the default
               model from settings.

    Returns:
        A ChatOpenRouter instance configured with the given model.
    """
    model = model or settings.default_model
    if model not in _cache:
        _cache[model] = ChatOpenRouter(
            model=model,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _cache[model]


def clear_cache() -> None:
    """Clear the LLM instance cache. Useful for testing."""
    _cache.clear()
