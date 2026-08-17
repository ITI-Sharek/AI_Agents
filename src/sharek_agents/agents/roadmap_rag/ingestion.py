"""Roadmap ingestion: chunk, embed and atomically persist roadmap chunks.

Flow per roadmap:

    1. chunk the roadmap text into meaningful pieces (order preserved)
    2. generate a validated 2048-dimensional embedding per chunk (reusing
       the shared embedding infrastructure)
    3. atomically replace the roadmap's chunks and advance ``processed_at``

The store and the embedding service are injected for testability.
"""

from __future__ import annotations

from sharek_agents.agents.roadmap_rag.chunking import chunk_roadmap
from sharek_agents.agents.roadmap_rag.embeddings import RoadmapEmbeddingService
from sharek_agents.agents.roadmap_rag.schemas import (
    ROADMAP_EMBEDDING_SCHEMA_VERSION,
    Roadmap,
    RoadmapChunkRecord,
)
from sharek_agents.agents.roadmap_rag.storage import RoadmapStorageError, RoadmapStore


# ── Error hierarchy ───────────────────────────────────────────────────────────


class RoadmapIngestionError(Exception):
    """Base error for roadmap ingestion."""


# ── Service ───────────────────────────────────────────────────────────────────


class RoadmapIngestionService:
    """Stores a roadmap definition and its embedded, ordered chunks."""

    def __init__(
        self,
        store: RoadmapStore,
        embedding_service: RoadmapEmbeddingService | None = None,
    ) -> None:
        self._store = store
        self._embedding_service = embedding_service or RoadmapEmbeddingService()

    async def ingest(
        self,
        title: str,
        roadmap_text: str,
        *,
        chunk_size: int | None = None,
    ) -> Roadmap:
        """Ingest one roadmap: definition + embedded ordered chunks.

        A blank roadmap text stores the definition with no chunks (and no
        embedding calls); a blank title is rejected. ``chunk_size`` is
        forwarded to the chunker (defaults to ``settings.chunk_size``);
        evaluation fixtures pass a small value so every ordered step is its
        own chunk and ``chunk_index`` equals the step order.

        Raises:
            RoadmapIngestionError: Blank title or storage failure.
            RoadmapEmbeddingError: Embedding generation/validation failed.
        """
        title = (title or "").strip()
        if not title:
            raise RoadmapIngestionError("roadmap title must not be empty")
        try:
            roadmap = await self._store.create_roadmap(title, roadmap_text)
        except RoadmapStorageError as exc:
            raise RoadmapIngestionError(
                f"Failed to store roadmap '{title}': {exc}"
            ) from exc

        await self._process_roadmap_definition(
            roadmap.id, roadmap_text, chunk_size=chunk_size
        )
        return roadmap

    async def process_existing_roadmap(
        self,
        roadmap_id: int,
        *,
        chunk_size: int | None = None,
    ) -> list[RoadmapChunkRecord]:
        """(Re)process an ALREADY stored roadmap: chunks + embeddings.

        Used by the automatic pre-retrieval synchronization flow for
        roadmaps inserted or updated directly in the ``roadmaps`` table.
        The current roadmap text is chunked, every chunk is embedded, the
        roadmap's old chunks are atomically replaced, and ``processed_at``
        is advanced — all in one atomic store operation. On any failure
        the old chunks and the old ``processed_at`` stay intact, so the
        roadmap remains stale and is safely retried later. Blank roadmap
        text stores no chunks (and makes no embedding calls) but is still
        marked processed. ``chunk_size`` is forwarded to the chunker
        (defaults to ``settings.chunk_size``).

        Raises:
            RoadmapIngestionError: The roadmap does not exist or chunk
                storage failed.
            RoadmapEmbeddingError: Embedding generation/validation failed.
        """
        roadmap = await self._store.get_roadmap(roadmap_id)
        if roadmap is None:
            raise RoadmapIngestionError(
                f"Roadmap {roadmap_id} does not exist; cannot process"
            )
        return await self._process_roadmap_definition(
            roadmap.id, roadmap.roadmap, chunk_size=chunk_size
        )

    async def _process_roadmap_definition(
        self,
        roadmap_id: int,
        roadmap_text: str,
        *,
        chunk_size: int | None = None,
    ) -> list[RoadmapChunkRecord]:
        """Chunk, embed and atomically persist one roadmap's chunks.

        All chunk embeddings are generated BEFORE any row is written, and
        the chunk replacement + ``processed_at`` update is one atomic
        store statement, so a failure never leaves a roadmap partially
        processed or marks it processed without complete chunks: it simply
        stays stale and is retried later.
        """
        chunks = chunk_roadmap(roadmap_text, chunk_size=chunk_size)
        records: list[RoadmapChunkRecord] = []
        for index, content in enumerate(chunks):
            records.append(
                RoadmapChunkRecord(
                    roadmap_id=roadmap_id,
                    chunk_index=index,
                    content=content,
                    embedding=await self._embedding_service.embed_text(content),
                    embedding_model=self._embedding_service.model,
                    embedding_model_version=self._embedding_service.model_version,
                    embedding_schema_version=ROADMAP_EMBEDDING_SCHEMA_VERSION,
                )
            )
        try:
            return await self._store.replace_chunks_and_set_processed_at(
                roadmap_id, records
            )
        except RoadmapStorageError as exc:
            raise RoadmapIngestionError(
                f"Failed to store roadmap chunks for roadmap {roadmap_id}: {exc}"
            ) from exc
