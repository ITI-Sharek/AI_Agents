"""Pre-retrieval roadmap synchronization (auto-ingestion + change detection).

Roadmaps are added and updated manually with SQL INSERT/UPDATE statements
directly against the existing ``roadmaps`` table. This component detects
roadmaps that need (re)processing — never processed (``processed_at`` is
NULL) or changed since their last successful processing (``updated_at >
processed_at``) — and processes them through the existing
``RoadmapIngestionService`` BEFORE retrieval answers any search, so the
Gap Guidance Agent never queries a stale knowledge base.

Responsibilities stay separated:

- synchronization: detect + process stale roadmaps;
- ``RoadmapIngestionService``: chunk + embed + atomically replace chunks +
  advance ``processed_at``;
- ``RealRoadmapRetriever`` / ``search_roadmap``: retrieval only.

The check runs on every Gap Guidance request, but only stale roadmaps are
actually processed; the normal case is one lightweight timestamp query.
Two layers prevent concurrent duplicate processing of the same roadmap:

- a process-local lock around the whole detect+process step;
- a session-scoped PostgreSQL advisory lock per roadmap (held on a
  dedicated pooled connection while chunking/embedding runs), with a
  staleness re-check inside it, so even concurrent requests in DIFFERENT
  processes never embed the same roadmap twice.
"""

from __future__ import annotations

import asyncio
import logging

from sharek_agents.agents.roadmap_rag.embeddings import RoadmapEmbeddingService
from sharek_agents.agents.roadmap_rag.ingestion import RoadmapIngestionService
from sharek_agents.agents.roadmap_rag.storage import RoadmapStore

logger = logging.getLogger(__name__)

#: Process-local lock shared by every synchronizer in this process, so two
#: concurrent Gap Guidance requests never both process the same roadmap.
_DEFAULT_LOCK = asyncio.Lock()


class RoadmapSynchronizationService:
    """Detect stale roadmaps and (re)process their chunks and embeddings."""

    def __init__(
        self,
        store: RoadmapStore,
        embedding_service: RoadmapEmbeddingService | None = None,
        *,
        lock: asyncio.Lock | None = None,
        chunk_size: int | None = None,
    ) -> None:
        self._store = store
        self._ingestion = RoadmapIngestionService(store, embedding_service)
        self._lock = lock if lock is not None else _DEFAULT_LOCK
        self._chunk_size = chunk_size

    async def synchronize(self) -> list[int]:
        """Process every currently stale roadmap; return their ids.

        One lightweight timestamp query detects roadmaps needing
        (re)processing; when none exist the method returns immediately.
        Each stale roadmap is processed under the shared process-local
        lock AND a session-scoped PostgreSQL advisory lock (with a
        staleness re-check inside it), so a concurrent request in the
        same or another process cannot embed it twice.

        Raises:
            RoadmapStorageError: The stale-detection query failed.
            RoadmapIngestionError: A roadmap is missing or its chunk
                storage failed.
            RoadmapEmbeddingError: Embedding generation/validation failed.
        """
        async with self._lock:
            stale = await self._store.get_stale_roadmaps()
            if not stale:
                return []
            processed: list[int] = []
            for roadmap in stale:
                async with await self._store.roadmap_processing_lock(
                    roadmap.id
                ):
                    if not await self._store.is_roadmap_stale(roadmap.id):
                        continue
                    logger.info(
                        "Processing stale roadmap %d (%s)",
                        roadmap.id,
                        roadmap.title,
                    )
                    await self._ingestion.process_existing_roadmap(
                        roadmap.id, chunk_size=self._chunk_size
                    )
                    processed.append(roadmap.id)
            logger.info("Processed %d stale roadmap(s)", len(processed))
            return processed


__all__ = ["RoadmapSynchronizationService"]
