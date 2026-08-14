"""LLM clients used by the Share-k AI workflows."""

import json
from typing import TypeVar

import httpx
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from sharek_agents.config import settings

_doc_understanding_cache: dict[str, ChatOpenAI] = {}
_openrouter_cache: dict[str, "OpenRouterLLM"] = {}
StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


def get_doc_understanding_llm() -> ChatOpenAI:
    """Get a cached LLM instance for the Documentation Understanding Agent.

    Reads provider, model, base URL, and API key from the
    ``doc_understanding_llm_*`` settings, keeping the ReAct Agent
    provider-configurable.

    Returns:
        A ChatOpenAI instance configured for the Documentation Understanding
        provider (e.g. Moonshot AI).
    """
    provider = settings.doc_understanding_llm_provider
    model = settings.doc_understanding_llm_model
    base_url = settings.doc_understanding_llm_base_url
    api_key = settings.doc_understanding_llm_api_key

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
    _doc_understanding_cache.clear()
    _openrouter_cache.clear()


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

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredModel],
    ) -> StructuredModel:
        """Generate and validate one JSON response from the text-only gateway."""

        schema = json.dumps(response_model.model_json_schema(by_alias=True))
        prompt = (
            f"{system_prompt}\n\n"
            "Return only valid JSON. Do not wrap it in Markdown or add "
            "explanatory text. The JSON must match this schema:\n"
            f"{schema}\n\n"
            f"{user_prompt}"
        )
        url = f"{self.base_url}/api/v1/student/chat"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model_id": self.model,
            "messages": [{"role": "user", "text": prompt}],
            "max_tokens": 1000,
            "temperature": 0.0,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, json=payload)

        if response.status_code == 401:
            raise RuntimeError("Invalid Student API Gateway API key.")
        if response.status_code == 403:
            raise RuntimeError(
                "Student API Gateway API key is not authorized "
                "to access this resource."
            )
        response.raise_for_status()

        raw_output = response.json().get("output_text")
        if not raw_output:
            raise RuntimeError(
                "Student API Gateway response is missing output_text."
            )
        return response_model.model_validate_json(_strip_json_fence(raw_output))


class OpenRouterLLM:
    """Small OpenRouter client with one explicit request and no hidden retries."""

    def __init__(
        self,
        model: str,
        api_key: str,
        timeout: float = 60.0,
    ):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = 0
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def invoke(self, prompt: str) -> str:
        """Generate plain text for workflows that own their own parser."""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 4000,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self.url,
                headers=self._headers,
                json=payload,
            )
        response.raise_for_status()
        return _openrouter_content(response.json())

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[StructuredModel],
    ) -> StructuredModel:
        """Request strict JSON Schema output and validate it with Pydantic."""

        schema = response_model.model_json_schema(by_alias=True)
        prompt_with_schema = (
            f"{user_prompt}\n\nReturn only JSON matching this schema; do not add prose or Markdown:\n"
            f"{json.dumps(schema)}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_with_schema},
            ],
            "temperature": 0.0,
            "max_tokens": 4000,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
            "provider": {"require_parameters": True},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.url,
                headers=self._headers,
                json=payload,
            )
            # Free OpenRouter routes do not all advertise JSON Schema support.
            # Retry only parameter-level rejection with broadly supported JSON
            # object mode; provider/auth failures surface immediately.
            if response.status_code in {400, 404, 422}:
                fallback = {**payload, "response_format": {"type": "json_object"}}
                fallback.pop("provider", None)
                response = await client.post(
                    self.url,
                    headers=self._headers,
                    json=fallback,
                )
        response.raise_for_status()
        raw_output = _openrouter_content(response.json())
        try:
            return response_model.model_validate_json(_strip_json_fence(raw_output))
        except (ValidationError, ValueError) as exc:
            raise RuntimeError("OpenRouter response did not match the requested schema") from exc


def _openrouter_content(payload: dict) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenRouter response is missing message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenRouter response is missing message content")
    return content


def _strip_json_fence(value: str) -> str:
    """Accept a JSON code fence defensively while still rejecting prose."""

    text = value.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _uses_student_gateway() -> bool:
    return settings.ai_provider.casefold() in {
        "student-api-gateway",
        "student_gateway",
        "getaway",
    }


def get_llm(model: str | None = None) -> OpenRouterLLM | StudentGatewayLLM:
    """Return the client selected by ``AI_PROVIDER``."""

    if not _uses_student_gateway():
        if settings.ai_provider.casefold() != "openrouter":
            raise RuntimeError(f"Unsupported AI provider: {settings.ai_provider}")
        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")

        model = model or settings.openrouter_model
        if model not in _openrouter_cache:
            _openrouter_cache[model] = OpenRouterLLM(
                model=model,
                api_key=settings.openrouter_api_key,
                timeout=settings.ai_skill_profile_timeout_seconds,
            )
        return _openrouter_cache[model]

    if not settings.getaway_iti_key:
        raise RuntimeError("GETAWAY_ITI_KEY is not configured")

    model = model or settings.getaway_model

    return StudentGatewayLLM(
        model=model,
        api_key=settings.getaway_iti_key,
        base_url=settings.getaway_base_url,
        timeout=settings.ai_skill_profile_timeout_seconds,
    )


async def generate_structured(
    *,
    system_prompt: str,
    user_prompt: str,
    response_model: type[StructuredModel],
) -> StructuredModel:
    """Generate validated output through the configured model provider."""

    return await get_llm().generate_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=response_model,
    )


def get_provider_metadata() -> tuple[str, str]:
    """Return truthful audit metadata for the configured provider."""

    if _uses_student_gateway():
        return "student-api-gateway", settings.getaway_model
    return "openrouter", settings.openrouter_model
