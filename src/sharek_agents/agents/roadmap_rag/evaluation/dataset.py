"""Ground-truth evaluation dataset for Roadmap RAG retrieval.

Every expected result is a ``(roadmap_id, chunk_index)`` reference into
``evaluation.fixtures.ROADMAP_FIXTURES`` — never an arbitrary value. The
fixtures are ingested with ``chunk_size=1`` so ``chunk_index`` equals the
step order, making the ground truth exact.

Coverage (see ``description``): exact skill match, skill + specific topic,
level-based gap, architecture gap, testing gap, multiple relevant chunks,
single-specific-chunk queries, irrelevant queries, missing roadmap
knowledge, queries requiring refinement, level + topic combinations, and
cross-roadmap topics.

Cases with an empty ``expected`` list are NO-KNOWLEDGE cases: nothing in
the fixtures is relevant, so a correct system must not return any
confident match (verified through the strong-match threshold, not through
retrieval metrics).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from sharek_agents.agents.roadmap_rag.evaluation.fixtures import (
    ROADMAP_FIXTURES,
    fixture_by_id,
)


class EvaluationCase(BaseModel):
    """One deterministic retrieval evaluation case with ground truth."""

    id: int
    skill: str
    query: str
    current_level: str | None = None
    target_level: str | None = None
    gap_description: str | None = None
    limit: int = Field(default=3, ge=1, le=10)
    expected: list[tuple[int, int]] = Field(
        default_factory=list,
        description="(roadmap_id, chunk_index) ground-truth chunk references",
    )
    refined_query: str | None = Field(
        default=None,
        description="Focused re-query used by the refinement case",
    )
    description: str = ""

    @model_validator(mode="after")
    def references_exist_in_fixtures(self) -> EvaluationCase:
        for roadmap_id, chunk_index in self.expected:
            fixture = fixture_by_id(roadmap_id)
            if not 0 <= chunk_index < len(fixture.steps):
                raise ValueError(
                    f"case {self.id}: fixture {roadmap_id} has "
                    f"{len(fixture.steps)} steps, expected chunk_index "
                    f"{chunk_index}"
                )
        return self

    @property
    def expected_roadmaps(self) -> set[int]:
        """Roadmap ids referenced by the expected chunks (nDCG grade-1 set)."""
        return {roadmap_id for roadmap_id, _ in self.expected}


EVALUATION_CASES: list[EvaluationCase] = [
    EvaluationCase(
        id=1,
        skill="postgresql",
        query="postgresql fundamentals",
        expected=[(2, 0)],
        description="exact skill match",
    ),
    EvaluationCase(
        id=2,
        skill="postgresql",
        query="postgresql indexing with EXPLAIN ANALYZE plans",
        expected=[(2, 1)],
        description="skill + specific topic",
    ),
    EvaluationCase(
        id=3,
        skill="fastapi",
        query="advanced fastapi production hardening",
        current_level="intermediate",
        target_level="advanced",
        gap_description="Level below required for production hardening",
        expected=[(1, 7)],
        description="level-based gap",
    ),
    EvaluationCase(
        id=4,
        skill="software architecture",
        query="clean architecture dependency inversion",
        expected=[(3, 3), (3, 2)],
        description="architecture-related gap (multiple expected chunks)",
    ),
    EvaluationCase(
        id=5,
        skill="fastapi",
        query="fastapi integration testing with the test client",
        expected=[(1, 5)],
        description="testing-related gap within a skill roadmap",
    ),
    EvaluationCase(
        id=6,
        skill="postgresql",
        query="postgresql indexing transactions query optimization",
        expected=[(2, 1), (2, 2), (2, 3)],
        description="multiple relevant roadmap chunks",
    ),
    EvaluationCase(
        id=7,
        skill="docker",
        query="docker secrets and environment configuration",
        expected=[(5, 3)],
        description="relevant information in one specific chunk",
    ),
    EvaluationCase(
        id=8,
        skill="postgresql",
        query="ancient greek philosophy rhetoric",
        expected=[],
        description="irrelevant query (no-knowledge case)",
    ),
    EvaluationCase(
        id=9,
        skill="quantum-computing",
        query="quantum circuit simulation",
        expected=[],
        description="missing roadmap knowledge (no-knowledge case)",
    ),
    EvaluationCase(
        id=10,
        skill="fastapi",
        query="fastapi overall improvement plan",
        refined_query="fastapi production hardening structured logging",
        expected=[(1, 7)],
        description="query requiring refinement (initial query misses)",
    ),
    EvaluationCase(
        id=11,
        skill="software architecture",
        query="scalability fault tolerance caching architecture",
        target_level="advanced",
        expected=[(3, 4)],
        description="level + topic combination",
    ),
    EvaluationCase(
        id=12,
        skill="testing",
        query="integration testing for databases and queues",
        expected=[(4, 2)],
        description="cross-roadmap topic",
    ),
]


def validate_dataset() -> None:
    """Validate every case against the fixtures (run at evaluation time)."""
    by_id: set[int] = set()
    for case in EVALUATION_CASES:
        if case.id in by_id:
            raise ValueError(f"duplicate evaluation case id {case.id}")
        by_id.add(case.id)
        case.references_exist_in_fixtures()
        if case.refined_query and case.refined_query.strip() == case.query.strip():
            raise ValueError(
                f"case {case.id}: refined_query must differ from query"
            )
    fixture_ids = {fixture.id for fixture in ROADMAP_FIXTURES}
    assert fixture_ids == {1, 2, 3, 4, 5}
