from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from sharek_agents.agents.document_understanding.chunker import DocumentChunk


# ── Error hierarchy ───────────────────────────────────────────────────────────


class VectorStoreError(Exception):
    """Base error for vector store operations."""


class IngestionError(VectorStoreError):
    """Raised when ingestion validation fails."""


class SearchError(VectorStoreError):
    """Raised when a search precondition is violated."""


# ── Search result ─────────────────────────────────────────────────────────────


@dataclass
class SearchResult:
    """A single search result with chunk and similarity score.

    Attributes:
        chunk: The matching document chunk.
        score: Cosine similarity score (0.0 to 1.0).
    """
    chunk: DocumentChunk
    score: float

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def filename(self) -> str | None:
        return self.chunk.filename

    @property
    def page_number(self) -> int | None:
        return self.chunk.page_number

    @property
    def section(self) -> str | None:
        return self.chunk.section

    @property
    def document_reference(self):
        return self.chunk.document_reference


# ── Protocol ──────────────────────────────────────────────────────────────────


class InMemoryStore(Protocol):
    """Request-scoped in-memory vector store.

    Populated at the start of a request and cleared after the
    response is produced. Never persisted to any database.
    """

    def ingest(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Store chunks together with their embedding vectors.

        Args:
            chunks: Document chunks to index.
            embeddings: Corresponding embedding vectors, one per chunk.

        Raises:
            IngestionError: If validation fails.
        """

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Return the top-k most similar chunks by cosine similarity.

        Args:
            query_embedding: The query vector.
            top_k: Maximum number of results to return.
            similarity_threshold: Minimum similarity score (0.0 to 1.0).

        Returns:
            List of SearchResult sorted by descending similarity.

        Raises:
            SearchError: If the query embedding is invalid.
        """

    def get_by_chunk_id(self, chunk_id: str) -> DocumentChunk | None:
        """Retrieve a chunk by its unique identifier.

        Args:
            chunk_id: The chunk identifier to look up.

        Returns:
            The matching chunk, or None if not found.
        """

    def clear(self) -> None:
        """Remove all entries (called at end of request)."""


# ── Cosine similarity ─────────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 for zero vectors or mismatched lengths.
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


# ── Internal entry ────────────────────────────────────────────────────────────


class MemoryStoreEntry:
    """A single entry in the in-memory vector store."""

    __slots__ = ("chunk", "embedding")

    def __init__(self, chunk: DocumentChunk, embedding: list[float]) -> None:
        self.chunk = chunk
        self.embedding = embedding


# ── Implementation ────────────────────────────────────────────────────────────


class VectorStore:
    """Request-scoped in-memory vector store with cosine similarity search.

    Each instance is isolated to a single request. Never shared between
    requests. Never persisted.

    Supports:
        - Ingesting chunks with embedding vectors.
        - Semantic similarity search with top-k and threshold.
        - Lookup by chunk ID.
        - Full cleanup via clear().
    """

    def __init__(self) -> None:
        self._entries: list[MemoryStoreEntry] = []
        self._chunk_index: dict[str, MemoryStoreEntry] = {}
        self._dimension: int | None = None

    def ingest(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            raise IngestionError("Chunks list must not be empty")
        if not embeddings:
            raise IngestionError("Embeddings list must not be empty")
        if len(chunks) != len(embeddings):
            raise IngestionError(
                f"Chunk/embedding count mismatch: "
                f"{len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        dim = len(embeddings[0])
        if dim == 0:
            raise IngestionError("Embedding vectors must have at least one dimension")

        if self._dimension is not None and dim != self._dimension:
            raise IngestionError(
                f"Vector dimension mismatch: store uses {self._dimension}, "
                f"got {dim}"
            )

        seen_ids: set[str] = set()
        for chunk, embedding in zip(chunks, embeddings):
            if not chunk.chunk_id:
                raise IngestionError("Chunk chunk_id must not be empty")
            if chunk.chunk_id in seen_ids:
                raise IngestionError(f"Duplicate chunk ID in batch: {chunk.chunk_id}")
            if chunk.chunk_id in self._chunk_index:
                raise IngestionError(
                    f"Chunk ID already exists in store: {chunk.chunk_id}"
                )
            if not embedding:
                raise IngestionError(
                    f"Empty embedding vector for chunk {chunk.chunk_id}"
                )
            if len(embedding) != dim:
                raise IngestionError(
                    f"Inconsistent vector dimension for chunk {chunk.chunk_id}: "
                    f"expected {dim}, got {len(embedding)}"
                )
            seen_ids.add(chunk.chunk_id)

        for chunk, embedding in zip(chunks, embeddings):
            entry = MemoryStoreEntry(chunk=chunk, embedding=embedding)
            self._entries.append(entry)
            self._chunk_index[chunk.chunk_id] = entry

        if self._dimension is None:
            self._dimension = dim

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> list[SearchResult]:
        if not query_embedding:
            raise SearchError("Query embedding must not be empty")

        if self._dimension is not None and len(query_embedding) != self._dimension:
            raise SearchError(
                f"Query embedding dimension {len(query_embedding)} "
                f"does not match store dimension {self._dimension}"
            )

        if not self._entries:
            return []

        results: list[SearchResult] = []
        for entry in self._entries:
            score = _cosine_similarity(query_embedding, entry.embedding)
            if score >= similarity_threshold:
                results.append(SearchResult(chunk=entry.chunk, score=score))

        results.sort(key=lambda r: (-r.score, r.chunk.chunk_id))

        return results[:top_k]

    def get_by_chunk_id(self, chunk_id: str) -> DocumentChunk | None:
        entry = self._chunk_index.get(chunk_id)
        return entry.chunk if entry else None

    def clear(self) -> None:
        self._entries.clear()
        self._chunk_index.clear()
        self._dimension = None

    @property
    def size(self) -> int:
        return len(self._entries)
