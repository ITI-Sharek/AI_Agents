"""In-memory ``RoadmapStore`` for deterministic evaluation.

Mirrors the SQL behavior of ``RoadmapPostgresStore`` (same record shape,
same upsert semantics, cosine similarity ranking with a deterministic tie
break, ``LIMIT`` truncation) so evaluation exercises the real retrieval
contract without requiring a live PostgreSQL database.
"""

from __future__ import annotations

import asyncio
import math
import types
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sharek_agents.agents.roadmap_rag.schemas import Roadmap, RoadmapChunkRecord


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity between two non-negative vectors (deterministic)."""
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return dot / (left_norm * right_norm)


class _MemoryLock:
    """Async context manager around an ``asyncio.Lock`` (advisory-lock stand-in)."""

    def __init__(self, lock: asyncio.Lock) -> None:
        self._lock = lock

    async def __aenter__(self) -> None:
        await self._lock.acquire()

    async def __aexit__(
        self,
        exc_type: object,
        exc: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        self._lock.release()


class InMemoryRoadmapStore:
    """Deterministic in-memory ``RoadmapStore`` implementation."""

    def __init__(self) -> None:
        self._roadmaps: dict[int, Roadmap] = {}
        self._chunks: dict[tuple[int, int], RoadmapChunkRecord] = {}
        self._next_roadmap_id = 1
        self._next_chunk_id = 1
        self._locks: dict[int, asyncio.Lock] = {}

    async def initialize(self) -> None:
        """No-op: no schema to create in memory."""

    async def create_roadmap(self, title: str, roadmap: str) -> Roadmap:
        now = _utcnow()
        record = Roadmap(
            id=self._next_roadmap_id,
            title=title,
            roadmap=roadmap,
            created_at=now,
            updated_at=now,
        )
        self._next_roadmap_id += 1
        self._roadmaps[record.id] = record
        return record

    async def store_chunks(
        self, chunks: Sequence[RoadmapChunkRecord]
    ) -> list[RoadmapChunkRecord]:
        stored: list[RoadmapChunkRecord] = []
        for chunk in chunks:
            existing = self._chunks.get((chunk.roadmap_id, chunk.chunk_index))
            if existing is not None:
                updated = existing.model_copy(update=chunk.model_dump())
                self._chunks[(chunk.roadmap_id, chunk.chunk_index)] = updated
                stored.append(updated)
            else:
                record = chunk.model_copy(
                    update={"id": self._next_chunk_id}
                )
                self._next_chunk_id += 1
                self._chunks[(record.roadmap_id, record.chunk_index)] = record
                stored.append(record)
        return stored

    async def get_roadmap(self, roadmap_id: int) -> Roadmap | None:
        return self._roadmaps.get(roadmap_id)

    async def get_stale_roadmaps(self) -> list[Roadmap]:
        return [
            roadmap
            for roadmap in sorted(self._roadmaps.values(), key=lambda r: r.id)
            if roadmap.processed_at is None
            or (
                roadmap.updated_at is not None
                and roadmap.processed_at is not None
                and roadmap.updated_at > roadmap.processed_at
            )
        ]

    async def is_roadmap_stale(self, roadmap_id: int) -> bool:
        roadmap = self._roadmaps.get(roadmap_id)
        if roadmap is None:
            return False
        return roadmap.processed_at is None or (
            roadmap.updated_at is not None
            and roadmap.processed_at is not None
            and roadmap.updated_at > roadmap.processed_at
        )

    async def replace_chunks_and_set_processed_at(
        self,
        roadmap_id: int,
        chunks: Sequence[RoadmapChunkRecord],
    ) -> list[RoadmapChunkRecord]:
        for (existing_roadmap_id, _chunk_index) in list(self._chunks):
            if existing_roadmap_id == roadmap_id:
                del self._chunks[(existing_roadmap_id, _chunk_index)]
        stored = await self.store_chunks(chunks)
        roadmap = self._roadmaps.get(roadmap_id)
        if roadmap is not None:
            self._roadmaps[roadmap_id] = roadmap.model_copy(
                update={"processed_at": _utcnow()}
            )
        return stored

    async def roadmap_processing_lock(self, roadmap_id: int) -> Any:
        return _MemoryLock(
            self._locks.setdefault(roadmap_id, asyncio.Lock())
        )

    async def search_chunks(
        self, query_embedding: list[float], limit: int
    ) -> list[dict[str, Any]]:
        scored: list[tuple[float, int, int, RoadmapChunkRecord]] = []
        for (roadmap_id, chunk_index), chunk in self._chunks.items():
            if chunk.embedding is None:
                continue
            roadmap = self._roadmaps.get(roadmap_id)
            if roadmap is None:
                continue
            similarity = _cosine_similarity(query_embedding, chunk.embedding)
            scored.append((similarity, roadmap_id, chunk_index, chunk))
        scored.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))
        return [
            {
                "roadmap_id": roadmap_id,
                "roadmap_title": self._roadmaps[roadmap_id].title,
                "chunk_index": chunk_index,
                "content": chunk.content,
                "cosine_similarity": similarity,
            }
            for similarity, roadmap_id, chunk_index, chunk in scored[:limit]
        ]

    def chunk_reference_lookup(self) -> dict[str, tuple[int, int]]:
        """Map every stored chunk content to its (roadmap_id, chunk_index).

        The ``RoadmapRetriever`` contract returns ``RoadmapChunk`` objects
        (skill/topic/content only), so evaluation resolves retrieved chunks
        back to ground-truth references through their unique content.
        """
        return {
            chunk.content: (roadmap_id, chunk_index)
            for (roadmap_id, chunk_index), chunk in self._chunks.items()
        }
