"""Roadmap knowledge storage: persistent roadmap + chunk repository.

The Roadmap database is KNOWLEDGE STORAGE, not Agent memory. This
repository is isolated from the existing ``PostgresSemanticMatchingStore``
(project/contributor behavior stays untouched) while reusing the SAME
PostgreSQL + pgvector database and the SAME migration mechanism and vector
codec wiring.

The store is driver-agnostic like the Semantic Matching store: it consumes
an injected "asyncpg-style" connection (``execute`` / ``fetchrow`` /
``fetch``), so tests can inject fakes and production injects the existing
``semantic_matching.database.create_connection_provider``.

Processing state is timestamp-based: ``roadmaps.processed_at`` (set only
after successful chunking + embedding) vs ``roadmaps.updated_at`` (bumped
by a database trigger when content changes). A roadmap is stale when
``processed_at IS NULL OR updated_at > processed_at``.
"""

from __future__ import annotations

import types
from typing import Any, Awaitable, Callable, Protocol, Sequence

from sharek_agents.agents.roadmap_rag.schemas import Roadmap, RoadmapChunkRecord
from sharek_agents.agents.semantic_matching.storage import migration_statements


# ── Error hierarchy ───────────────────────────────────────────────────────────


class RoadmapStorageError(Exception):
    """Base error for roadmap storage operations."""


class RoadmapStorageConfigurationError(RoadmapStorageError):
    """The roadmap storage was used without a configured database connection."""


#: Advisory-lock namespace constant ("RMP"): roadmap processing locks are
#: derived as ``namespace << 32 | roadmap_id`` so they never collide with
#: locks used elsewhere.
_ROADMAP_LOCK_NAMESPACE = 0x524D50


def _roadmap_lock_key(roadmap_id: int) -> int:
    """Stable session-scoped advisory-lock key for one roadmap."""
    return (_ROADMAP_LOCK_NAMESPACE << 32) | (roadmap_id & 0xFFFFFFFF)


# ── Protocol ──────────────────────────────────────────────────────────────────


class RoadmapStore(Protocol):
    """Persistent roadmap knowledge repository (roadmaps + roadmap_chunks)."""

    async def initialize(self) -> None:
        """Apply the database migrations (idempotent)."""

    async def create_roadmap(self, title: str, roadmap: str) -> Roadmap:
        """Store a roadmap definition and return the created record."""

    async def store_chunks(
        self, chunks: Sequence[RoadmapChunkRecord]
    ) -> list[RoadmapChunkRecord]:
        """Insert or refresh chunk records (with embeddings)."""

    async def get_roadmap(self, roadmap_id: int) -> Roadmap | None:
        """Retrieve a stored roadmap definition by id."""

    async def get_stale_roadmaps(self) -> list[Roadmap]:
        """Return roadmaps that need (re)processing.

        A roadmap is stale when it was never processed (``processed_at``
        is NULL) or its content changed after the last successful
        processing (``updated_at > processed_at``). Results are ordered by
        id.
        """

    async def is_roadmap_stale(self, roadmap_id: int) -> bool:
        """Whether one roadmap still needs (re)processing right now."""

    async def replace_chunks_and_set_processed_at(
        self,
        roadmap_id: int,
        chunks: Sequence[RoadmapChunkRecord],
    ) -> list[RoadmapChunkRecord]:
        """Atomically replace one roadmap's chunks and mark it processed.

        Deletes the roadmap's old chunks, inserts the new chunks, and
        advances ``processed_at`` in ONE statement: if any part fails the
        whole operation rolls back (old chunks and the old ``processed_at``
        stay intact, so the roadmap remains stale and retryable).
        """

    async def roadmap_processing_lock(self, roadmap_id: int) -> Any:
        """Acquire a session-scoped advisory lock for one roadmap.

        Returns an async context manager that releases the lock on exit.
        Used so concurrent requests never embed the same stale roadmap
        twice, even across processes.
        """

    async def search_chunks(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Rank chunks by cosine similarity to a query embedding.

        Each item is ``{"roadmap_id", "roadmap_title", "chunk_index",
        "content", "cosine_similarity"}``; results are ordered by cosine
        similarity descending and truncated to ``limit``. Empty when no
        chunk has a stored embedding.
        """


# ── PostgreSQL + pgvector implementation ──────────────────────────────────────


class RoadmapPostgresStore:
    """PostgreSQL + pgvector roadmap knowledge repository.

    The database connection is injected (the existing asyncpg-style
    duck-type connection), so no driver dependency is required in this
    repository; statements use asyncpg-style ``$1`` placeholders and
    ``::vector`` casts for pgvector compatibility, mirroring the Semantic
    Matching store.
    """

    _ROADMAP_COLUMNS = "id, title, roadmap, created_at, updated_at, processed_at"
    _CHUNK_COLUMNS = (
        "roadmap_id, chunk_index, content, embedding, embedding_model, "
        "embedding_model_version, embedding_schema_version, "
        "created_at, updated_at"
    )

    def __init__(
        self,
        connection_provider: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        self._connection_provider = connection_provider

    def _connection(self) -> Any:
        if self._connection_provider is None:
            raise RoadmapStorageConfigurationError(
                "No database connection configured for roadmap storage. "
                "Configure SEMANTIC_MATCHING_DATABASE_URL and inject a "
                "connection provider."
            )
        return self._connection_provider()

    async def initialize(self) -> None:
        connection = await self._connection()
        for statement in migration_statements():
            await connection.execute(statement)

    async def create_roadmap(self, title: str, roadmap: str) -> Roadmap:
        connection = await self._connection()
        row = await connection.fetchrow(
            "INSERT INTO roadmaps (title, roadmap) VALUES ($1, $2) "
            f"RETURNING {self._ROADMAP_COLUMNS}",
            title,
            roadmap,
        )
        return self._row_to_roadmap(row)

    async def store_chunks(
        self, chunks: Sequence[RoadmapChunkRecord]
    ) -> list[RoadmapChunkRecord]:
        if not chunks:
            return []
        connection = await self._connection()
        values: list[str] = []
        args: list[Any] = []
        for chunk in chunks:
            offset = len(args)
            args.extend(
                [
                    chunk.roadmap_id,
                    chunk.chunk_index,
                    chunk.content,
                    chunk.embedding,
                    chunk.embedding_model,
                    chunk.embedding_model_version,
                    chunk.embedding_schema_version,
                ]
            )
            values.append(
                "("
                + ", ".join(
                    [
                        f"${offset + 1}",
                        f"${offset + 2}",
                        f"${offset + 3}",
                        f"${offset + 4}::vector",
                        f"${offset + 5}",
                        f"${offset + 6}",
                        f"${offset + 7}",
                        "now()",
                        "now()",
                    ]
                )
                + ")"
            )
        rows = await connection.fetch(
            f"""
            INSERT INTO roadmap_chunks ({self._CHUNK_COLUMNS})
            VALUES {", ".join(values)}
            ON CONFLICT (roadmap_id, chunk_index) DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                embedding_model = EXCLUDED.embedding_model,
                embedding_model_version = EXCLUDED.embedding_model_version,
                embedding_schema_version = EXCLUDED.embedding_schema_version,
                updated_at = now()
            RETURNING id, {self._CHUNK_COLUMNS}
            """,
            *args,
        )
        return [self._row_to_chunk(row) for row in rows]

    async def get_roadmap(self, roadmap_id: int) -> Roadmap | None:
        connection = await self._connection()
        row = await connection.fetchrow(
            f"SELECT {self._ROADMAP_COLUMNS} FROM roadmaps WHERE id = $1",
            roadmap_id,
        )
        return self._row_to_roadmap(row) if row is not None else None

    async def get_stale_roadmaps(self) -> list[Roadmap]:
        connection = await self._connection()
        rows = await connection.fetch(
            f"""
            SELECT {self._ROADMAP_COLUMNS}
            FROM roadmaps
            WHERE processed_at IS NULL OR updated_at > processed_at
            ORDER BY id
            """
        )
        return [self._row_to_roadmap(row) for row in rows]

    async def is_roadmap_stale(self, roadmap_id: int) -> bool:
        connection = await self._connection()
        row = await connection.fetchrow(
            """
            SELECT EXISTS (
                SELECT 1
                FROM roadmaps
                WHERE id = $1
                  AND (processed_at IS NULL OR updated_at > processed_at)
            ) AS exists
            """,
            roadmap_id,
        )
        return bool(row["exists"])

    async def replace_chunks_and_set_processed_at(
        self,
        roadmap_id: int,
        chunks: Sequence[RoadmapChunkRecord],
    ) -> list[RoadmapChunkRecord]:
        connection = await self._connection()
        if not chunks:
            await connection.execute(
                """
                WITH replaced AS (
                    DELETE FROM roadmap_chunks WHERE roadmap_id = $1
                ), marked AS (
                    UPDATE roadmaps SET processed_at = now() WHERE id = $1
                )
                SELECT 1
                """,
                roadmap_id,
            )
            return []
        values: list[str] = []
        args: list[Any] = [roadmap_id]
        for chunk in chunks:
            offset = len(args)
            args.extend(
                [
                    chunk.roadmap_id,
                    chunk.chunk_index,
                    chunk.content,
                    chunk.embedding,
                    chunk.embedding_model,
                    chunk.embedding_model_version,
                    chunk.embedding_schema_version,
                ]
            )
            values.append(
                "("
                + ", ".join(
                    [
                        f"${offset + 1}",
                        f"${offset + 2}",
                        f"${offset + 3}",
                        f"${offset + 4}::vector",
                        f"${offset + 5}",
                        f"${offset + 6}",
                        f"${offset + 7}",
                        "now()",
                        "now()",
                    ]
                )
                + ")"
            )
        rows = await connection.fetch(
            f"""
            WITH replaced AS (
                DELETE FROM roadmap_chunks WHERE roadmap_id = $1
            ), inserted AS (
                INSERT INTO roadmap_chunks ({self._CHUNK_COLUMNS})
                VALUES {", ".join(values)}
                ON CONFLICT (roadmap_id, chunk_index) DO NOTHING
                RETURNING id, {self._CHUNK_COLUMNS}
            ), marked AS (
                UPDATE roadmaps SET processed_at = now() WHERE id = $1
            )
            SELECT id, {self._CHUNK_COLUMNS} FROM inserted
            """,
            *args,
        )
        return [self._row_to_chunk(row) for row in rows]

    async def roadmap_processing_lock(self, roadmap_id: int) -> Any:
        connection = await self._connection()
        raw = await connection.acquire()
        key = _roadmap_lock_key(roadmap_id)
        try:
            await raw.fetchrow("SELECT pg_advisory_lock($1)", key)
        except BaseException:
            await connection.release(raw)
            raise
        return _RoadmapProcessingLock(
            connection=raw,
            release_callback=connection.release,
            key=key,
        )

    async def search_chunks(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        connection = await self._connection()
        rows = await connection.fetch(
            """
            SELECT c.roadmap_id, r.title AS roadmap_title, c.chunk_index,
                   c.content,
                   1 - (c.embedding <=> $1::vector) AS cosine_similarity
            FROM roadmap_chunks c
            JOIN roadmaps r ON r.id = c.roadmap_id
            WHERE c.embedding IS NOT NULL
            ORDER BY cosine_similarity DESC
            LIMIT $2
            """,
            query_embedding,
            limit,
        )
        return [
            {
                "roadmap_id": row["roadmap_id"],
                "roadmap_title": row["roadmap_title"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "cosine_similarity": float(row["cosine_similarity"]),
            }
            for row in rows
        ]

    @staticmethod
    def _row_to_roadmap(row: Any) -> Roadmap:
        return Roadmap(
            id=row["id"],
            title=row["title"],
            roadmap=row["roadmap"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            processed_at=row["processed_at"],
        )

    @staticmethod
    def _row_to_chunk(row: Any) -> RoadmapChunkRecord:
        embedding = row["embedding"]
        return RoadmapChunkRecord(
            id=row["id"],
            roadmap_id=row["roadmap_id"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            embedding=list(embedding) if embedding is not None else None,
            embedding_model=row["embedding_model"],
            embedding_model_version=row["embedding_model_version"],
            embedding_schema_version=row["embedding_schema_version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class _RoadmapProcessingLock:
    """Session-scoped PostgreSQL advisory lock for one roadmap.

    The advisory lock is held on a dedicated pooled connection acquired by
    the store, so it spans ALL of the roadmap's processing work (chunking
    and embedding happen outside the database). Exiting the context
    releases the lock and returns the connection to the pool.
    """

    def __init__(
        self,
        *,
        connection: Any,
        release_callback: Callable[[Any], Awaitable[None]],
        key: int,
    ) -> None:
        self._connection = connection
        self._release_callback = release_callback
        self._key = key

    async def __aenter__(self) -> None:
        """The lock was already acquired before this context started."""

    async def __aexit__(
        self,
        exc_type: object,
        exc: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        try:
            await self._connection.fetchrow(
                "SELECT pg_advisory_unlock($1)", self._key
            )
        finally:
            await self._release_callback(self._connection)


# ── Factory ───────────────────────────────────────────────────────────────────


def create_roadmap_store(
    connection_provider: Callable[[], Awaitable[Any]] | None = None,
) -> RoadmapPostgresStore:
    """Build the PostgreSQL + pgvector roadmap knowledge repository."""
    return RoadmapPostgresStore(connection_provider=connection_provider)
