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
    model = model or settings.openrouter_model
    if model not in _cache:
        _cache[model] = ChatOpenRouter(
            model=model,
            api_key=settings.openrouter_api_key,
            # timeout is in milliseconds; convert from seconds
            timeout=int(settings.ai_skill_profile_timeout_seconds * 1000),
            temperature=0.0,
        )
    return _cache[model]


def clear_cache() -> None:
    """Clear the LLM instance cache. Useful for testing."""
    _cache.clear()

'''


"""Student API Gateway LLM client."""

"""Student API Gateway LLM client."""

import httpx

from sharek_agents.config import settings


class StudentGatewayLLM:
    """Client for the Student API Gateway."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 60.0,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def invoke(self, prompt: str) -> str:
        """Send a text prompt to the Student API Gateway."""

        url = f"{self.base_url}/api/v1/student/chat"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model_id": self.model,
            "messages": [
                {
                    "role": "user",
                    "text": prompt,
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.0,
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                url,
                headers=headers,
                json=payload,
            )

        if response.status_code == 401:
            raise RuntimeError(
                "Invalid Student API Gateway API key."
            )

        if response.status_code == 403:
            raise RuntimeError(
                "Student API Gateway API key is not authorized "
                "to access this resource."
            )

        response.raise_for_status()

        data = response.json()

        output_text = data.get("output_text")

        if not output_text:
            raise RuntimeError(
                "Student API Gateway response is missing output_text."
            )

        return output_text


def get_llm(model: str | None = None) -> StudentGatewayLLM:
    """Get an LLM client configured for the Student API Gateway."""

    model = model or settings.getaway_model

    return StudentGatewayLLM(
        model=model,
        api_key=settings.getaway_iti_key,
        base_url=settings.getaway_base_url,
        timeout=settings.ai_skill_profile_timeout_seconds,
    )

'''