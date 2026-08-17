"""Deterministic lexical embeddings for Roadmap RAG evaluation.

The evaluation must be reproducible and must not require the real embedding
provider, so these embeddings replace the HTTP client with a deterministic
bag-of-words hash: tokens (word unigrams and adjacent bigrams) are hashed
with ``zlib.crc32`` into the same 2048-dimension space the production
``vector(2048)`` column uses. Cosine similarity over these vectors is a
stable, meaningful lexical-overlap measure, so retrieval ranking is fully
deterministic while still exercising the real retrieval code path
(``RealRoadmapRetriever`` -> search text -> one query embedding -> cosine
ranking -> top-K).

The class duck-types ``document_understanding.embeddings.EmbeddingService``
and is injected through ``RoadmapEmbeddingService``.
"""

from __future__ import annotations

import re
import zlib

from sharek_agents.agents.semantic_matching.storage import EMBEDDING_DIMENSIONS

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    words = _WORD_RE.findall(text.casefold())
    bigrams = [
        f"{words[index]} {words[index + 1]}"
        for index in range(len(words) - 1)
    ]
    return words + bigrams


def deterministic_vector(text: str) -> list[float]:
    """Build a deterministic 2048-d bag-of-words vector for ``text``."""
    vector = [0.0] * EMBEDDING_DIMENSIONS
    for token in _tokens(text):
        vector[zlib.crc32(token.encode("utf-8")) % EMBEDDING_DIMENSIONS] += 1.0
    return vector


class DeterministicEmbeddingService:
    """In-memory deterministic ``EmbeddingService`` for evaluation/tests.

    ``embed_query_calls`` counts query embeddings so tests can verify the
    retriever generates exactly one query embedding per search.
    """

    def __init__(self) -> None:
        self.embed_query_calls = 0

    @property
    def dimensions(self) -> int:
        return EMBEDDING_DIMENSIONS

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [deterministic_vector(text) for text in texts]

    async def embed_query(self, query: str) -> list[float]:
        self.embed_query_calls += 1
        return deterministic_vector(query)
