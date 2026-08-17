"""Real roadmap retrieval backend for the Gap Guidance Agent.

``RealRoadmapRetriever`` satisfies the existing
``gap_guidance.retrieval.RoadmapRetriever`` protocol (the same interface
the ``search_roadmap`` tool already consumes) and answers retrieval
requests with the persistent Roadmap RAG infrastructure:

    skill + query + levels + gap description
            |
            v
    focused search text
            |
            v
    RoadmapEmbeddingService (ONE query embedding, 2048-d)
            |
            v
    roadmap_chunks vector search (cosine similarity, HNSW when available)
            |
            v
    most relevant RoadmapChunk objects (respecting ``limit``)

This is PURELY retrieval: it never decides whether a contributor has a
gap, never generates guidance, and never calls an LLM.
"""

from __future__ import annotations

from sharek_agents.agents.gap_guidance.retrieval import RoadmapChunk
from sharek_agents.agents.roadmap_rag.embeddings import (
    RoadmapEmbeddingError,
    RoadmapEmbeddingService,
)
from sharek_agents.agents.roadmap_rag.storage import (
    RoadmapStorageError,
    RoadmapStore,
    create_roadmap_store,
)
from sharek_agents.agents.semantic_matching.database import (
    create_connection_provider,
)
from sharek_agents.config import settings


# ── Error hierarchy ───────────────────────────────────────────────────────────


class RoadmapRetrievalError(Exception):
    """The roadmap retrieval backend could not answer a search request."""


# ── Retrieval ─────────────────────────────────────────────────────────────────


def build_search_text(
    skill: str,
    query: str,
    current_level: str | None = None,
    target_level: str | None = None,
    gap_description: str | None = None,
) -> str:
    """Build the focused search text embedded for the vector query."""
    parts: list[str] = [skill, query]
    if current_level:
        parts.append(f"current level: {current_level}")
    if target_level:
        parts.append(f"target level: {target_level}")
    if gap_description:
        parts.append(gap_description)
    return "\n".join(part for part in parts if part)


class RealRoadmapRetriever:
    """Vector-search roadmap retrieval over ``roadmap_chunks``.

    The store and embedding service are injected for testability.
    """

    def __init__(
        self,
        store: RoadmapStore,
        embedding_service: RoadmapEmbeddingService | None = None,
    ) -> None:
        self._store = store
        self._embedding_service = embedding_service or RoadmapEmbeddingService()

    async def search(
        self,
        *,
        skill: str,
        query: str,
        current_level: str | None = None,
        target_level: str | None = None,
        gap_description: str | None = None,
        limit: int = 3,
    ) -> list[RoadmapChunk]:
        search_text = build_search_text(
            skill, query, current_level, target_level, gap_description
        )
        try:
            query_embedding = await self._embedding_service.embed_text(search_text)
        except RoadmapEmbeddingError as exc:
            raise RoadmapRetrievalError(
                f"Roadmap retrieval could not generate the query embedding: {exc}"
            ) from exc
        try:
            results = await self._store.search_chunks(query_embedding, limit)
        except RoadmapStorageError as exc:
            raise RoadmapRetrievalError(
                f"Roadmap vector search failed: {exc}"
            ) from exc
        return [
            RoadmapChunk(
                skill=skill,
                topic=item["roadmap_title"],
                content=item["content"],
            )
            for item in results
        ]


# ── Factory ───────────────────────────────────────────────────────────────────


def create_real_roadmap_retriever(
    store: RoadmapStore | None = None,
    embedding_service: RoadmapEmbeddingService | None = None,
) -> RealRoadmapRetriever:
    """Build the production roadmap retriever.

    ``store`` / ``embedding_service`` override the production defaults
    (used by tests). The production store reuses the existing Semantic
    Matching PostgreSQL + pgvector connection wiring
    (``SEMANTIC_MATCHING_DATABASE_URL``); when the URL is not configured the
    store is created without a connection and fails with a clear
    ``RoadmapStorageConfigurationError`` at search time instead of
    inventing values.
    """
    if store is None:
        database_url = settings.semantic_matching_database_url
        connection_provider = (
            create_connection_provider(database_url) if database_url else None
        )
        store = create_roadmap_store(connection_provider)
    return RealRoadmapRetriever(store=store, embedding_service=embedding_service)
