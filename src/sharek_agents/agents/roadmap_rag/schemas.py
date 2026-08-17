"""Roadmap RAG schemas: persistent roadmap knowledge records.

The Roadmap database is KNOWLEDGE STORAGE, not Agent memory. It stores
roadmap definitions, chunks, and their embeddings only — never
conversations, contributor memory, project memory, Agent state, previous
reasoning, or previous queries.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


#: Version of the roadmap chunk embedding schema (metadata recorded with
#: every stored vector so vectors can be detected and regenerated when the
#: schema changes, mirroring the Semantic Matching convention).
ROADMAP_EMBEDDING_SCHEMA_VERSION = "1"


class Roadmap(BaseModel):
    """One stored roadmap definition (``roadmaps`` table row).

    ``processed_at`` records the last successful chunking + embedding
    processing; a roadmap is stale when ``processed_at`` is NULL or
    ``updated_at`` is newer than it.
    """

    id: int
    title: str
    roadmap: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    processed_at: datetime | None = None


class RoadmapChunkRecord(BaseModel):
    """One stored roadmap chunk with its embedding (``roadmap_chunks`` row).

    ``chunk_index`` preserves the roadmap step/order information.
    Embedding metadata mirrors the Semantic Matching convention
    (``embedding_model`` / ``embedding_model_version`` / 
    ``embedding_schema_version``).
    """

    id: int | None = Field(default=None, description="Assigned by the database")
    roadmap_id: int
    chunk_index: int = Field(ge=0)
    content: str
    embedding: list[float] | None = None
    embedding_model: str | None = None
    embedding_model_version: str | None = None
    embedding_schema_version: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
