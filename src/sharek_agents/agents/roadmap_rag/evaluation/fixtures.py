"""Deterministic Roadmap RAG evaluation fixtures.

Each fixture is a realistic roadmap with clearly identifiable ordered
steps. The evaluation ingests these fixtures through the REAL chunking and
ingestion code with ``chunk_size=1``, so every step becomes exactly one
chunk and ``chunk_index`` equals the step order. Ground truth in the
evaluation dataset is expressed as ``(roadmap_id, chunk_index)`` references
derived from THESE fixtures, never arbitrary values.

Step texts deliberately repeat the skill/topic vocabulary the evaluation
queries use, so the deterministic lexical embeddings produce meaningful,
reproducible cosine rankings.
"""

from __future__ import annotations

from pydantic import BaseModel


class RoadmapFixture(BaseModel):
    """One deterministic evaluation roadmap."""

    id: int
    title: str
    steps: list[str]


ROADMAP_FIXTURES: list[RoadmapFixture] = [
    RoadmapFixture(
        id=1,
        title="FastAPI Roadmap",
        steps=[
            "FastAPI fundamentals: learn HTTP semantics and request lifecycle.",
            "FastAPI dependency injection: master dependencies and composition.",
            "FastAPI middleware: compose middleware and ASGI applications.",
            "FastAPI authentication: add token and session based auth.",
            "FastAPI authorization: enforce roles and permissions on routes.",
            "FastAPI testing: write unit tests with pytest and integration tests with the test client.",
            "FastAPI deployment: deploy with uvicorn workers and reverse proxies.",
            "FastAPI production hardening: add structured logging, rate limiting, and graceful shutdown.",
        ],
    ),
    RoadmapFixture(
        id=2,
        title="PostgreSQL Roadmap",
        steps=[
            "PostgreSQL fundamentals: learn schema basics, data types, and constraints.",
            "PostgreSQL indexing: read EXPLAIN ANALYZE plans and design covering indexes.",
            "PostgreSQL transactions: practice isolation levels and concurrency control.",
            "PostgreSQL query optimization: tune slow queries and join strategies.",
            "PostgreSQL operations: tune vacuum, autovacuum, and connection pooling.",
        ],
    ),
    RoadmapFixture(
        id=3,
        title="Software Architecture Roadmap",
        steps=[
            "Software architecture fundamentals: reason about components and boundaries.",
            "Layered architecture: separate presentation, business, and data layers.",
            "Dependency inversion: depend on abstractions, not implementations.",
            "Clean architecture: apply hexagonal or clean architecture to a real project.",
            "Scalability: reason about horizontal scaling, caching, and fault tolerance.",
            "System design: practice trade-off analysis and capacity planning.",
        ],
    ),
    RoadmapFixture(
        id=4,
        title="Testing Roadmap",
        steps=[
            "Testing fundamentals: build a test pyramid with fast unit tests.",
            "Unit testing: write focused unit tests for pure logic.",
            "Integration testing: cover boundaries such as databases and queues with test doubles.",
            "End to end testing: add e2e tests for critical user paths.",
            "Test driven development: practice TDD on new features.",
        ],
    ),
    RoadmapFixture(
        id=5,
        title="Docker Roadmap",
        steps=[
            "Docker fundamentals: build minimal images and layers.",
            "Docker multi stage builds: separate build and runtime images.",
            "Docker compose: run multi service stacks with health checks.",
            "Docker secrets: manage secrets and environment configuration.",
            "Docker deployment: push images to a registry and deploy containers.",
        ],
    ),
]


def fixture_by_id(roadmap_id: int) -> RoadmapFixture:
    """Return the fixture with the given id (used by ground truth helpers)."""
    for fixture in ROADMAP_FIXTURES:
        if fixture.id == roadmap_id:
            return fixture
    raise ValueError(f"no roadmap fixture with id {roadmap_id}")
