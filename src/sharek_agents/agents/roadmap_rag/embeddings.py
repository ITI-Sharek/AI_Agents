"""Roadmap embedding generation, reusing the Semantic Matching infrastructure.

No new embedding provider, HTTP client, or model is created here: the
dedicated ``ROADMAP_RAG_EMBEDDING_*`` settings select the same repository
embedding factory (``document_understanding.embeddings.create_embedding_service``)
when set, and fall back to the Semantic Matching embedding client
(``semantic_matching/llm.py`` factory) when they are missing. The service
validates that every returned vector matches the pgvector column dimension
(``vector(2048)``), exactly like the Semantic Matching embedding adapter.
Embedding metadata (``model`` / ``model_version``) is recorded consistently
with the Semantic Matching convention.
"""

from __future__ import annotations

from sharek_agents.agents.document_understanding.embeddings import (
    EmbeddingError,
    EmbeddingService,
    create_embedding_service,
)
from sharek_agents.agents.semantic_matching.llm import (
    SemanticMatchingEmbeddingConfigError,
    get_semantic_matching_embedding_client,
)
from sharek_agents.agents.semantic_matching.storage import EMBEDDING_DIMENSIONS
from sharek_agents.config import settings

# ── Error hierarchy ───────────────────────────────────────────────────────────


class RoadmapEmbeddingError(Exception):
    """Base error for roadmap embedding generation."""


class RoadmapEmbeddingConfigurationError(RoadmapEmbeddingError):
    """The shared embedding provider/model/API key configuration is missing."""


class RoadmapEmbeddingGenerationError(RoadmapEmbeddingError):
    """The embedding provider failed to produce a vector."""


class RoadmapEmbeddingDimensionMismatchError(RoadmapEmbeddingError):
    """The returned vector dimension does not match the pgvector column."""


# ── Dedicated embedding client factory ────────────────────────────────────────


_EMBEDDING_CACHE: dict[str, EmbeddingService] = {}


def get_roadmap_rag_embedding_client() -> EmbeddingService:
    """Get the dedicated embedding client for Roadmap RAG.

    Dedicated mode is active when both ``ROADMAP_RAG_EMBEDDING_PROVIDER``
    and ``ROADMAP_RAG_EMBEDDING_MODEL`` are set; the dedicated API key and
    base URL (``ROADMAP_RAG_EMBEDDING_API_KEY`` /
    ``ROADMAP_RAG_EMBEDDING_BASE_URL``) are passed to the repository's
    existing embedding factory (``create_embedding_service``), never
    hardcoded, never logged. Instances are cached per
    provider/model/base-URL combination.

    When the dedicated provider or model is missing, the client falls back
    to the existing Semantic Matching embedding configuration
    (``get_semantic_matching_embedding_client``), preserving the previous
    runtime behavior exactly.
    """
    provider = settings.roadmap_rag_embedding_provider
    model = settings.roadmap_rag_embedding_model
    if not provider or not model:
        return get_semantic_matching_embedding_client()

    api_key = settings.roadmap_rag_embedding_api_key
    base_url = settings.roadmap_rag_embedding_base_url

    cache_key = f"{provider}:{model}:{base_url}"
    if cache_key not in _EMBEDDING_CACHE:
        try:
            _EMBEDDING_CACHE[cache_key] = create_embedding_service(
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url or None,
                timeout_seconds=settings.embedding_timeout_seconds,
            )
        except EmbeddingError as exc:
            raise RoadmapEmbeddingConfigurationError(
                f"Roadmap embedding configuration is invalid: {exc}"
            ) from exc
    return _EMBEDDING_CACHE[cache_key]


def clear_embedding_client_cache() -> None:
    """Clear the dedicated client cache. Useful for testing."""
    _EMBEDDING_CACHE.clear()


# ── Service ───────────────────────────────────────────────────────────────────


class RoadmapEmbeddingService:
    """Generates pgvector-compatible embeddings from plain text.

    Wraps the dedicated Roadmap RAG embedding client (created lazily
    through ``get_roadmap_rag_embedding_client``) and validates that every
    returned vector matches the pgvector column dimension
    (``storage.EMBEDDING_DIMENSIONS``) before it is handed to the roadmap
    storage layer.
    """

    def __init__(self, service: EmbeddingService | None = None) -> None:
        self._service = service

    @property
    def model(self) -> str:
        """The configured embedding model identifier."""
        return settings.embedding_model

    @property
    def model_version(self) -> str:
        """Deterministic model version (the configured model identifier)."""
        return self.model

    def _resolved_service(self) -> EmbeddingService:
        """Return the dedicated embedding client, built once on first use.

        Creation is lazy so an unconfigured embedding setup only fails when
        embedding is actually requested, not when the service is wired.
        """
        if self._service is None:
            try:
                self._service = get_roadmap_rag_embedding_client()
            except SemanticMatchingEmbeddingConfigError as exc:
                raise RoadmapEmbeddingConfigurationError(
                    f"Roadmap embedding configuration is invalid: {exc}"
                ) from exc
        return self._service

    async def embed_text(self, text: str) -> list[float]:
        """Generate the validated embedding vector for one chunk/query text.

        Raises:
            RoadmapEmbeddingConfigurationError: Missing/invalid embedding
                configuration.
            RoadmapEmbeddingGenerationError: Provider failure or invalid
                response.
            RoadmapEmbeddingDimensionMismatchError: Vector dimension != 2048.
        """
        service = self._resolved_service()
        try:
            vector = await service.embed_query(text)
        except EmbeddingError as exc:
            raise RoadmapEmbeddingGenerationError(
                f"Roadmap embedding generation failed: {exc}"
            ) from exc

        if len(vector) != EMBEDDING_DIMENSIONS:
            raise RoadmapEmbeddingDimensionMismatchError(
                f"Embedding vector has {len(vector)} dimensions but the "
                f"roadmap_chunks.embedding column requires "
                f"{EMBEDDING_DIMENSIONS} (model: {self.model})."
            )
        return vector
