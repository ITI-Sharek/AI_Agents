"""Roadmap retrieval backend for the Gap Guidance Agent.

Phase 2 ships a deterministic mock retrieval backend only: no database, no
vector index, no embeddings, no ingestion pipeline. ``MockRoadmapRetriever``
matches the requested skill and query against a small static in-memory
roadmap knowledge base and returns the best-scoring chunks.

The Agent never talks to this module directly: it only sees the
``search_roadmap`` tool. The ``RoadmapRetriever`` protocol is the seam where
a real RAG implementation can be plugged in later without changing the tool
or the Agent.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RoadmapChunk(BaseModel):
    """One retrieved roadmap knowledge chunk returned to the Agent.

    ``content`` is a compact roadmap segment (progression steps for the
    topic); the Agent must reason over these chunks rather than inventing
    resources on its own.
    """

    skill: str = Field(description="The skill the chunk belongs to")
    topic: str = Field(description="Short topic label for the chunk")
    content: str = Field(description="Roadmap knowledge text for the topic")


class RoadmapRetriever(Protocol):
    """Retrieval interface used by ``search_roadmap``.

    The real RAG implementation must implement the same protocol; the Agent
    tool contract stays unchanged.
    """

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
        """Return the most relevant roadmap chunks for the request."""
        ...


_ROADMAP_KNOWLEDGE: tuple[dict[str, object], ...] = (
    {
        "skill": "fastapi",
        "topic": "advanced-fastapi",
        "keywords": ["fastapi", "asgi", "middleware", "dependencies", "openapi", "async"],
        "content": (
            "FastAPI progression: master dependency injection and middleware "
            "composition first, then move to async database sessions and "
            "background task patterns. Practice OpenAPI-driven API design with "
            "request validation, then add caching, rate limiting, and "
            "production hardening such as structured logging and graceful "
            "shutdown."
        ),
    },
    {
        "skill": "fastapi",
        "topic": "fastapi-testing",
        "keywords": ["fastapi", "testing", "pytest", "httpx", "testclient"],
        "content": (
            "FastAPI testing path: write unit tests for routers with "
            "dependency overrides, add integration tests with the test client "
            "against a real database, then cover error paths and "
            "authentication before moving to property-based tests for "
            "request/response validation."
        ),
    },
    {
        "skill": "postgresql",
        "topic": "postgresql-optimization",
        "keywords": ["postgresql", "indexes", "explain", "query", "plan", "performance"],
        "content": (
            "PostgreSQL progression: learn to read EXPLAIN ANALYZE plans, "
            "design covering indexes for the slowest queries, then practice "
            "transaction isolation levels and concurrency control before "
            "tuning vacuum, autovacuum, and connection pooling."
        ),
    },
    {
        "skill": "postgresql",
        "topic": "postgresql-schema-design",
        "keywords": ["postgresql", "schema", "normalization", "migrations", "constraints"],
        "content": (
            "PostgreSQL schema path: practice normalization and constraint "
            "design, write versioned migrations with rollback plans, then "
            "move to advanced modeling such as partitioning, JSONB, and "
            "full-text search indexes."
        ),
    },
    {
        "skill": "docker",
        "topic": "docker-production",
        "keywords": ["docker", "compose", "multi-stage", "image", "container", "deployment"],
        "content": (
            "Docker progression: start with multi-stage builds and minimal "
            "base images, then compose multi-service stacks with health checks "
            "and volumes, and finish with image scanning, secret handling, and "
            "registry-based deployment pipelines."
        ),
    },
    {
        "skill": "react",
        "topic": "react-performance",
        "keywords": ["react", "hooks", "state", "memoization", "rendering", "components"],
        "content": (
            "React progression: master hooks and state management patterns "
            "first, then learn memoization and rendering optimization, and "
            "finish with server-side rendering or static generation and "
            "client-side performance budgets."
        ),
    },
    {
        "skill": "software architecture",
        "topic": "clean-architecture",
        "keywords": ["architecture", "clean", "layers", "dependency", "inversion", "hexagonal", "design"],
        "content": (
            "Architecture progression: practice dependency inversion and "
            "layered boundaries in small services first, then apply "
            "hexagonal or clean architecture to a real project, and finally "
            "reason about scalability, fault tolerance, and system design "
            "trade-offs."
        ),
    },
    {
        "skill": "testing",
        "topic": "testing-strategy",
        "keywords": ["testing", "unit", "integration", "coverage", "test", "pytest"],
        "content": (
            "Testing progression: build a test pyramid with fast unit tests, "
            "add integration tests for boundaries such as databases and "
            "queues, then add end-to-end tests for critical paths and "
            "practice test-driven development on new features."
        ),
    },
    {
        "skill": "redis",
        "topic": "redis-caching",
        "keywords": ["redis", "cache", "pubsub", "streams", "caching"],
        "content": (
            "Redis progression: learn key design and cache invalidation "
            "strategies first, then practice pub/sub and streams for "
            "background processing, and finish with persistence, eviction "
            "policies, and high-availability topologies."
        ),
    },
    {
        "skill": "kubernetes",
        "topic": "kubernetes-foundations",
        "keywords": ["kubernetes", "pods", "deployments", "services", "helm", "containers"],
        "content": (
            "Kubernetes progression: start with pods, deployments, and "
            "services, then practice configuration with ConfigMaps and "
            "secrets, and finish with ingress, horizontal autoscaling, "
            "resource limits, and observability with probes and metrics."
        ),
    },
    {
        "skill": "graphql",
        "topic": "graphql-api-design",
        "keywords": ["graphql", "schema", "resolvers", "queries", "mutations", "federation"],
        "content": (
            "GraphQL progression: design schemas and resolvers with proper "
            "data-loading first, then handle caching, batching, and "
            "authorization, and finish with federation and schema "
            "governance for larger teams."
        ),
    },
)


def _normalize(text: str) -> str:
    return text.casefold()


def _tokens(text: str) -> set[str]:
    return {token for token in _normalize(text).replace("-", " ").split() if token}


class MockRoadmapRetriever:
    """Deterministic in-memory retrieval backend for Phase 2.

    Scores the static knowledge chunks by token overlap between the request
    (query + skill) and each chunk's keywords, with a boost for chunks whose
    skill matches the requested skill exactly. Results are ordered by score
    descending then knowledge order, and truncated to ``limit``. A request
    with no overlap returns no chunks — the Agent must reason over partial
    or empty retrieval, exactly as it will with the real RAG backend.
    """

    def __init__(self, knowledge: Sequence[dict[str, object]] | None = None) -> None:
        self._knowledge = list(knowledge) if knowledge is not None else list(_ROADMAP_KNOWLEDGE)

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
        request_skill = _normalize(skill.strip())
        request_tokens = _tokens(query)

        scored: list[tuple[int, int, dict[str, object]]] = []
        for index, chunk in enumerate(self._knowledge):
            keywords = {_normalize(str(keyword)) for keyword in chunk["keywords"]}
            score = 0
            for token in request_tokens:
                for keyword in keywords:
                    if token == keyword:
                        score += 2
                    elif len(token) >= 4 and token in keyword:
                        score += 1
            if request_skill and _normalize(str(chunk["skill"])) == request_skill:
                score += 3
            if score > 0:
                scored.append((score, index, chunk))

        scored.sort(key=lambda entry: (-entry[0], entry[1]))

        return [
            RoadmapChunk(
                skill=str(chunk["skill"]),
                topic=str(chunk["topic"]),
                content=str(chunk["content"]),
            )
            for _, _, chunk in scored[:limit]
        ]