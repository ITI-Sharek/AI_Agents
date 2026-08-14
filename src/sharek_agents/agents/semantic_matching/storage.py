"""Semantic Matching storage layer (Phase 1 revision: pgvector).

Independent matching index over PostgreSQL + pgvector, separate from the
main business database. The main database remains the source of truth; the
Semantic Matching feature never writes to it.

Phases 1-7 provide the storage foundation:

- ``initialize`` applies the pgvector migration.
- ``get_*`` / ``upsert_*`` read and write matching-index records, including
  freshness metadata and embedding metadata.
- ``find_projects_by_similarity`` ranks indexed Projects by cosine
  similarity against a query vector (Phase 8 production matching).

Ranking and matching behavior live in the service layer; the store only
answers the vector query. Records may exist without an embedding until
embedding generation runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from sharek_agents.agents.semantic_matching.schemas import (
    ContributorMatchRecord,
    ProjectMatchRecord,
)


# ── Error hierarchy ───────────────────────────────────────────────────────────


class MatchingStorageError(Exception):
    """Base error for semantic matching storage operations."""


class MatchingStorageConfigurationError(MatchingStorageError):
    """The storage was used without a configured database connection."""


# ── Migration setup ───────────────────────────────────────────────────────────


MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
INITIAL_SCHEMA = MIGRATIONS_DIR / "001_initial_schema.sql"

#: pgvector column dimension used in the initial schema.
EMBEDDING_DIMENSIONS = 2048


def _migration_statements() -> list[str]:
    """Split the migration file into individual executable statements.

    ``--`` comment text is removed first so semicolons inside comments
    (e.g. ``-- ... source of truth; this ...``) cannot split a statement.
    """
    text = "\n".join(
        line.split("--", 1)[0] for line in INITIAL_SCHEMA.read_text(encoding="utf-8").splitlines()
    )
    statements: list[str] = []
    for raw in text.split(";"):
        statement = raw.strip()
        if statement:
            statements.append(statement + ";")
    return statements


# ── Protocol ──────────────────────────────────────────────────────────────────


class SemanticMatchingStore(Protocol):
    """Internal matching index for the Semantic Matching feature.

    Accessed only through the service/repository layer; never exposed as an
    HTTP CRUD API. Records are identified by their original entity ids
    (``project_id`` / ``contributor_id``) only.
    """

    async def initialize(self) -> None:
        """Apply the schema/migrations (idempotent)."""

    async def upsert_project(self, record: ProjectMatchRecord) -> ProjectMatchRecord:
        """Insert or refresh a Project matching-index record."""

    async def upsert_contributor(
        self, record: ContributorMatchRecord
    ) -> ContributorMatchRecord:
        """Insert or refresh a Contributor matching-index record."""

    async def get_project(self, project_id: int) -> ProjectMatchRecord | None:
        """Retrieve a Project matching-index record by ``project_id``."""

    async def get_contributor(self, contributor_id: int) -> ContributorMatchRecord | None:
        """Retrieve a Contributor matching-index record by ``contributor_id``."""

    async def find_projects_by_similarity(
        self,
        query_embedding: list[float],
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rank indexed Projects by cosine similarity to a query embedding.

        All Projects with a stored embedding are returned, ordered by
        cosine similarity descending; ``limit`` truncates the result.
        Each item is ``{"project_id": int, "cosine_similarity": float}``.
        """


# ── Helpers ───────────────────────────────────────────────────────────────────


def _jsonb(items: list[Any]) -> str:
    return json.dumps([item.model_dump(mode="json") for item in items])


def _decode_jsonb(value: Any) -> list[Any]:
    if isinstance(value, str):
        return json.loads(value)
    return value or []


def _decode_embedding(value: Any) -> list[float] | None:
    if value is None:
        return None
    return list(value)


# ── PostgreSQL + pgvector implementation ──────────────────────────────────────


class PostgresSemanticMatchingStore:
    """PostgreSQL + pgvector matching index.

    The database connection is injected (an asyncpg-style connection or
    pool ``execute``/``fetchrow``/``fetch`` duck-type) so no driver
    dependency is required in this repository; statements use asyncpg-style
    ``$1`` placeholders and ``::vector`` casts for pgvector compatibility.
    """

    _PROJECT_COLUMNS = (
        "project_id, skills, evidence, source_version, source_updated_at, "
        "embedding, embedding_model, embedding_model_version, "
        "embedding_schema_version, created_at, updated_at"
    )
    _CONTRIBUTOR_COLUMNS = (
        "contributor_id, skills, evidence, source_version, source_updated_at, "
        "embedding, embedding_model, embedding_model_version, "
        "embedding_schema_version, created_at, updated_at"
    )

    def __init__(
        self,
        connection_provider: Callable[[], Awaitable[Any]] | None = None,
    ) -> None:
        self._connection_provider = connection_provider

    def _connection(self) -> Any:
        if self._connection_provider is None:
            raise MatchingStorageConfigurationError(
                "No database connection configured for the semantic matching "
                "index. Configure SEMANTIC_MATCHING_DATABASE_URL and inject a "
                "connection provider in a later phase."
            )
        return self._connection_provider()

    async def initialize(self) -> None:
        connection = await self._connection()
        for statement in _migration_statements():
            await connection.execute(statement)

    async def upsert_project(self, record: ProjectMatchRecord) -> ProjectMatchRecord:
        connection = await self._connection()
        row = await connection.fetchrow(
            f"""
            INSERT INTO projects ({self._PROJECT_COLUMNS})
            VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8, $9, now(), now())
            ON CONFLICT (project_id) DO UPDATE SET
                skills = EXCLUDED.skills,
                evidence = EXCLUDED.evidence,
                source_version = EXCLUDED.source_version,
                source_updated_at = EXCLUDED.source_updated_at,
                embedding = EXCLUDED.embedding,
                embedding_model = EXCLUDED.embedding_model,
                embedding_model_version = EXCLUDED.embedding_model_version,
                embedding_schema_version = EXCLUDED.embedding_schema_version,
                updated_at = now()
            RETURNING {self._PROJECT_COLUMNS}
            """,
            record.project_id,
            _jsonb(record.skills),
            _jsonb(record.evidence),
            record.source_version,
            record.source_updated_at,
            record.embedding,
            record.embedding_model,
            record.embedding_model_version,
            record.embedding_schema_version,
        )
        return self._row_to_project(row)

    async def upsert_contributor(
        self, record: ContributorMatchRecord
    ) -> ContributorMatchRecord:
        connection = await self._connection()
        row = await connection.fetchrow(
            f"""
            INSERT INTO contributors ({self._CONTRIBUTOR_COLUMNS})
            VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8, $9, now(), now())
            ON CONFLICT (contributor_id) DO UPDATE SET
                skills = EXCLUDED.skills,
                evidence = EXCLUDED.evidence,
                source_version = EXCLUDED.source_version,
                source_updated_at = EXCLUDED.source_updated_at,
                embedding = EXCLUDED.embedding,
                embedding_model = EXCLUDED.embedding_model,
                embedding_model_version = EXCLUDED.embedding_model_version,
                embedding_schema_version = EXCLUDED.embedding_schema_version,
                updated_at = now()
            RETURNING {self._CONTRIBUTOR_COLUMNS}
            """,
            record.contributor_id,
            _jsonb(record.skills),
            _jsonb(record.evidence),
            record.source_version,
            record.source_updated_at,
            record.embedding,
            record.embedding_model,
            record.embedding_model_version,
            record.embedding_schema_version,
        )
        return self._row_to_contributor(row)

    async def get_project(self, project_id: int) -> ProjectMatchRecord | None:
        connection = await self._connection()
        row = await connection.fetchrow(
            f"SELECT {self._PROJECT_COLUMNS} FROM projects WHERE project_id = $1",
            project_id,
        )
        return self._row_to_project(row) if row is not None else None

    async def get_contributor(self, contributor_id: int) -> ContributorMatchRecord | None:
        connection = await self._connection()
        row = await connection.fetchrow(
            f"SELECT {self._CONTRIBUTOR_COLUMNS} FROM contributors "
            "WHERE contributor_id = $1",
            contributor_id,
        )
        return self._row_to_contributor(row) if row is not None else None

    async def find_projects_by_similarity(
        self,
        query_embedding: list[float],
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        connection = await self._connection()
        sql = (
            "SELECT project_id, 1 - (embedding <=> $1::vector) AS cosine_similarity "
            "FROM projects "
            "WHERE embedding IS NOT NULL "
            "ORDER BY cosine_similarity DESC"
        )
        if limit is not None:
            sql += " LIMIT $2"
            rows = await connection.fetch(sql, query_embedding, limit)
        else:
            rows = await connection.fetch(sql, query_embedding)
        return [
            {
                "project_id": row["project_id"],
                "cosine_similarity": float(row["cosine_similarity"]),
            }
            for row in rows
        ]

    @staticmethod
    def _row_to_project(row: Any) -> ProjectMatchRecord:
        return ProjectMatchRecord(
            project_id=row["project_id"],
            skills=_decode_jsonb(row["skills"]),
            evidence=_decode_jsonb(row["evidence"]),
            source_version=row["source_version"],
            source_updated_at=row["source_updated_at"],
            embedding=_decode_embedding(row["embedding"]),
            embedding_model=row["embedding_model"],
            embedding_model_version=row["embedding_model_version"],
            embedding_schema_version=row["embedding_schema_version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_contributor(row: Any) -> ContributorMatchRecord:
        return ContributorMatchRecord(
            contributor_id=row["contributor_id"],
            skills=_decode_jsonb(row["skills"]),
            evidence=_decode_jsonb(row["evidence"]),
            source_version=row["source_version"],
            source_updated_at=row["source_updated_at"],
            embedding=_decode_embedding(row["embedding"]),
            embedding_model=row["embedding_model"],
            embedding_model_version=row["embedding_model_version"],
            embedding_schema_version=row["embedding_schema_version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# ── Factory ───────────────────────────────────────────────────────────────────


def create_matching_store(
    connection_provider: Callable[[], Awaitable[Any]] | None = None,
) -> PostgresSemanticMatchingStore:
    """Build the PostgreSQL + pgvector matching index.

    ``settings.semantic_matching_database_url`` is the configured connection
    target; the actual driver/connection injection is supplied by the
    deployment wiring (later phase). Pass ``connection_provider`` in
    environments that already own a database connection.
    """
    return PostgresSemanticMatchingStore(connection_provider=connection_provider)