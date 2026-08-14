"""Semantic Matching embedding generation (Phase 3).

Phase 1 fixed the pgvector column dimension (``vector(2048)``) and Phase 2
defined the canonical representation (``build_embedding_input``). Phase 3
adds the embedding service that turns that canonical text into a validated
vector:

    Project/Contributor source data
            |
            v
    build_embedding_input()  (Phase 2 canonical representation)
            |
            v
    SemanticMatchingEmbeddingService --embed()--> list[float]
            |
            v
    pgvector (vector(2048))

This module resolves its embedding client ONLY through the dedicated
``semantic_matching/llm.py`` factory (``get_semantic_matching_embedding_client``),
which builds the repository's OpenAI-compatible ``EmbeddingService``
(``document_understanding/embeddings.py``) with the OpenRouter
configuration from ``sharek_agents.config``. This module is a thin
SEMANTIC MATCHING adapter only, adding:

- the canonical-text entry point (Phase 2 representation), and
- validation that the returned vector matches the pgvector column dimension.

Embedding model (decision): the repository's existing embedding convention,
``nvidia/nemotron-3-embed-1b:free`` (OpenRouter), whose native output
dimension is 2048 — exactly the Phase 1 ``vector(2048)`` column dimension, so
no migration change is needed.

Model version (decision): the embedding providers used here (OpenRouter /
OpenAI-compatible endpoints) do not expose a model version in their
responses. The safest deterministic value is therefore the configured model
identifier itself: ``embedding_model`` and ``embedding_model_version`` both
record the model slug, so any configured model change is automatically
detected as a version change by later phases.

This module does NOT know how matching works: no similarity, ranking, or
search behavior exists here.
"""

from __future__ import annotations

from sharek_agents.agents.document_understanding.embeddings import EmbeddingService
from sharek_agents.agents.semantic_matching.llm import (
    SemanticMatchingEmbeddingConfigError,
    get_semantic_matching_embedding_client,
)
from sharek_agents.agents.semantic_matching.representation import build_embedding_input
from sharek_agents.agents.semantic_matching.schemas import (
    ContributorSourceData,
    ProjectSourceData,
)
from sharek_agents.agents.semantic_matching.storage import EMBEDDING_DIMENSIONS
from sharek_agents.config import settings


# ── Error hierarchy ───────────────────────────────────────────────────────────


class SemanticMatchingEmbeddingError(Exception):
    """Base error for Semantic Matching embedding generation."""


class EmbeddingConfigurationError(SemanticMatchingEmbeddingError):
    """Embedding provider/model/API key configuration is missing or invalid."""


class EmbeddingGenerationError(SemanticMatchingEmbeddingError):
    """The embedding provider failed (timeout, auth, rate limit, empty/invalid response)."""


class EmbeddingDimensionMismatchError(SemanticMatchingEmbeddingError):
    """The returned vector dimension does not match the pgvector column."""


# ── Service ───────────────────────────────────────────────────────────────────


class SemanticMatchingEmbeddingService:
    """Generates pgvector-compatible embeddings from canonical text.

    Wraps the dedicated ``semantic_matching/llm.py`` embedding client
    (created lazily through ``get_semantic_matching_embedding_client``)
    and validates that every returned vector matches the pgvector column
    dimension (``storage.EMBEDDING_DIMENSIONS``) before it is handed to
    the storage layer.
    """

    def __init__(
        self,
        service: EmbeddingService | None = None,
        provider: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._service = service
        self._provider = provider
        self._model = model
        self._timeout_seconds = timeout_seconds

    @property
    def model(self) -> str:
        """The configured embedding model identifier."""
        return self._model or settings.embedding_model

    @property
    def model_version(self) -> str:
        """Deterministic model version.

        The embedding providers used here do not expose a model version in
        their responses, so the configured model identifier itself is the
        safest deterministic version value: any model change is detected as
        a version change.
        """
        return self.model

    def _resolved_service(self) -> EmbeddingService:
        """Return the dedicated embedding client, built once on first use.

        Creation is lazy so an unconfigured embedding setup only fails when
        embedding is actually requested, not when the service is wired.
        """
        if self._service is None:
            try:
                self._service = get_semantic_matching_embedding_client(
                    provider=self._provider,
                    model=self._model,
                    timeout_seconds=self._timeout_seconds,
                )
            except SemanticMatchingEmbeddingConfigError as exc:
                raise EmbeddingConfigurationError(
                    f"Semantic Matching embedding configuration is invalid: {exc}"
                ) from exc
        return self._service

    async def embed(
        self,
        data: ProjectSourceData | ContributorSourceData,
    ) -> list[float]:
        """Generate the embedding vector for an entity's canonical text.

        Flow (Phase 2 -> Phase 3):

            SourceData
                |
                v
            build_embedding_input(data)
                |
                v
            embedding provider
                |
                v
            validated vector (len == storage.EMBEDDING_DIMENSIONS)

        Args:
            data: The authoritative Project or Contributor source data.

        Returns:
            The validated embedding vector, ready for pgvector storage.

        Raises:
            EmbeddingConfigurationError: Missing/invalid embedding configuration.
            EmbeddingGenerationError: Provider failure or invalid response.
            EmbeddingDimensionMismatchError: Vector dimension != pgvector column.
        """
        service = self._resolved_service()
        text = build_embedding_input(data)
        try:
            vector = await service.embed_query(text)
        except EmbeddingError as exc:
            raise EmbeddingGenerationError(
                f"Embedding generation failed: {exc}"
            ) from exc

        if len(vector) != EMBEDDING_DIMENSIONS:
            raise EmbeddingDimensionMismatchError(
                f"Embedding vector has {len(vector)} dimensions but the "
                f"pgvector matching index requires {EMBEDDING_DIMENSIONS} "
                f"(model: {self.model}). Regenerate the vector with a "
                f"matching model."
            )
        return vector


# ── Factory ───────────────────────────────────────────────────────────────────


def create_semantic_matching_embedding_service(
    service: EmbeddingService | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
) -> SemanticMatchingEmbeddingService:
    """Build the Semantic Matching embedding service.

    ``service``/``provider``/``model``/``timeout_seconds`` override the
    dedicated ``semantic_matching/llm.py`` configuration when provided
    (the underlying dedicated factory raises if no API key is
    configured).
    """
    return SemanticMatchingEmbeddingService(
        service=service,
        provider=provider,
        model=model,
        timeout_seconds=timeout_seconds,
    )