from functools import lru_cache

from langchain_openrouter import ChatOpenRouter

from sharek_agents.config import settings


@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenRouter(
        model=settings.default_model,
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )
