from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import httpx

from sharek_agents.config import settings


# ── Error hierarchy ───────────────────────────────────────────────────────────


class EmbeddingError(Exception):
    """Base error for embedding generation failures."""


class EmbeddingProviderError(EmbeddingError):
    """The embedding provider returned an error response."""


class EmbeddingTimeoutError(EmbeddingError):
    """The embedding request timed out."""


class EmbeddingRateLimitError(EmbeddingError):
    """The embedding provider rate-limited the request."""


class EmbeddingEmptyResultError(EmbeddingError):
    """The provider returned an empty result set."""


class EmbeddingDimensionError(EmbeddingError):
    """Embedding dimensions are inconsistent or unexpected."""


class EmbeddingAuthError(EmbeddingError):
    """Authentication with the embedding provider failed."""


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class EmbeddingConfig:
    """Provider-agnostic configuration for embedding generation."""

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: str = ""
    base_url: str = "https://api.openai.com"
    dimensions: int | None = None
    timeout_seconds: float = 30.0


# ── Protocol ──────────────────────────────────────────────────────────────────


class EmbeddingService(Protocol):
    """Provider-agnostic interface for generating text embeddings.

    The provider and model are configured at construction time
    via settings, never hard-coded.
    """

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input text, in the same order.

        Raises:
            EmbeddingError: If embedding generation fails.
        """

    async def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single query text.

        Args:
            query: The query string to embed.

        Returns:
            A single embedding vector.

        Raises:
            EmbeddingError: If embedding generation fails.
        """

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of generated embedding vectors."""
        ...


# ── HTTP-based implementation (OpenAI-compatible) ────────────────────────────


class HttpEmbeddingService:
    """Embedding service using any OpenAI-compatible HTTP API.

    Works with OpenAI, OpenRouter, and other providers that expose
    the ``POST /v1/embeddings`` interface.

    The HTTP client can be injected for testability. When not provided,
    an ``httpx.AsyncClient`` is created and owned by this instance.
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._http_client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout_seconds)
        )
        self._owns_client = http_client is None
        self._base_url = config.base_url.rstrip("/")
        self._dimensions: int | None = None

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning vectors in the same order."""
        if not texts:
            return []

        valid_indices = [i for i, t in enumerate(texts) if t]
        if not valid_indices:
            return self._make_zero_vectors(len(texts))

        valid_texts = [texts[i] for i in valid_indices]
        vectors = await self._call_api(valid_texts)

        if len(valid_indices) < len(texts):
            full: list[list[float] | None] = [None] * len(texts)
            for idx, vec in zip(valid_indices, vectors):
                full[idx] = vec
            dim = self.dimensions
            for i, t in enumerate(texts):
                if not t:
                    full[i] = [0.0] * dim
            vectors = [v for v in full if v is not None]

        return vectors

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query text and return its vector."""
        if not query:
            raise EmbeddingError("Query text must not be empty.")
        results = await self.embed_texts([query])
        return results[0]

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            raise EmbeddingError(
                "No embeddings generated yet. Call embed_texts first "
                "or configure dimensions explicitly."
            )
        return self._dimensions

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        payload: dict = {
            "input": texts,
            "model": self._config.model,
        }
        if self._config.dimensions is not None:
            payload["dimensions"] = self._config.dimensions

        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await self._http_client.post(
                f"{self._base_url}/v1/embeddings",
                headers=headers,
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise EmbeddingTimeoutError(
                f"Embedding request timed out after {self._config.timeout_seconds}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingProviderError(
                f"HTTP error during embedding request: {exc}"
            ) from exc

        if response.status_code == 429:
            raise EmbeddingRateLimitError(
                "Embedding provider returned HTTP 429 (rate limited)"
            )
        if response.status_code == 401:
            raise EmbeddingAuthError(
                "Embedding provider returned HTTP 401 (unauthorized). "
                "Check that the API key is valid."
            )
        if response.status_code >= 400:
            body = _safe_response_text(response)
            raise EmbeddingProviderError(
                f"Embedding provider returned HTTP {response.status_code}: {body}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise EmbeddingProviderError(
                "Embedding provider returned non-JSON response"
            ) from exc

        if not isinstance(data, dict) or "data" not in data:
            raise EmbeddingProviderError(
                "Embedding provider response is missing 'data' field"
            )

        items = data["data"]
        if not items:
            raise EmbeddingEmptyResultError(
                "Embedding provider returned an empty data array"
            )

        items.sort(key=lambda item: item.get("index", 0))

        vectors: list[list[float]] = []
        for item in items:
            emb = item.get("embedding")
            if emb is None:
                raise EmbeddingProviderError(
                    "Embedding provider response item is missing 'embedding' field"
                )
            vectors.append(emb)

        if len(vectors) != len(texts):
            raise EmbeddingDimensionError(
                f"Expected {len(texts)} vectors, got {len(vectors)}"
            )

        dim = len(vectors[0])
        for vec in vectors:
            if len(vec) != dim:
                raise EmbeddingDimensionError(
                    f"Inconsistent vector dimensions in response: "
                    f"expected {dim}, got {len(vec)}"
                )

        if self._dimensions is None:
            self._dimensions = dim
        elif dim != self._dimensions:
            raise EmbeddingDimensionError(
                f"Vector dimension changed: was {self._dimensions}, now {dim}"
            )

        return vectors

    def _make_zero_vectors(self, count: int) -> list[list[float]]:
        if self._dimensions is None:
            raise EmbeddingError(
                "Cannot create zero vectors: embedding dimensions unknown. "
                "Configure dimensions explicitly or ensure at least one "
                "non-empty text has been embedded first."
            )
        return [[0.0] * self._dimensions for _ in range(count)]

    async def close(self) -> None:
        """Close the underlying HTTP client if owned by this instance."""
        if self._owns_client:
            await self._http_client.aclose()


# ── Factory ───────────────────────────────────────────────────────────────────


_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com",
    "openrouter": "https://openrouter.ai/api",
}


def _resolve_embedding_api_key(provider: str) -> str:
    """Resolve the API key for the given provider.

    Uses the explicit embedding API key first, then falls back
    to provider-specific keys.
    """
    explicit = settings.embedding_api_key
    if explicit:
        return explicit
    if provider == "openrouter":
        return settings.openrouter_api_key
    return ""


def create_embedding_service(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    dimensions: int | None = None,
    timeout_seconds: float | None = None,
) -> EmbeddingService:
    """Create an ``EmbeddingService`` from configuration.

    When arguments are ``None`` they are read from the global settings
    singleton.  At minimum a valid API key must be available either via
    arguments, settings, or provider-specific fallback.
    """
    resolved_provider = provider or settings.embedding_provider
    resolved_model = model or settings.embedding_model
    resolved_api_key = (
        api_key
        if api_key is not None
        else _resolve_embedding_api_key(resolved_provider)
    )
    resolved_base_url: str
    if base_url is not None:
        resolved_base_url = base_url
    elif settings.doc_understanding_embedding_base_url:
        resolved_base_url = settings.doc_understanding_embedding_base_url
    else:
        resolved_base_url = _PROVIDER_BASE_URLS.get(resolved_provider, "")
    resolved_dimensions = (
        dimensions
        if dimensions is not None
        else settings.embedding_dimensions
    )
    resolved_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else settings.embedding_timeout_seconds
    )

    cfg = EmbeddingConfig(
        provider=resolved_provider,
        model=resolved_model,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        dimensions=resolved_dimensions,
        timeout_seconds=resolved_timeout,
    )

    if not cfg.api_key:
        raise EmbeddingError(
            f"API key not configured for embedding provider '{cfg.provider}'. "
            f"Set DOC_UNDERSTANDING_EMBEDDING_API_KEY or the provider-specific "
            f"environment variable."
        )

    if not cfg.base_url:
        raise EmbeddingError(
            f"No base URL known for embedding provider '{cfg.provider}'. "
            f"Set DOC_UNDERSTANDING_EMBEDDING_BASE_URL or register the provider."
        )

    return HttpEmbeddingService(config=cfg)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _safe_response_text(response: httpx.Response) -> str:
    """Extract error text from a response without leaking secrets."""
    try:
        body = response.text
        if len(body) > 200:
            body = body[:200] + "..."
        return body
    except Exception:
        return "<unreadable response body>"
