"""RAG retrieval metrics over the deterministic evaluation dataset.

Metric definitions (documented explicitly so every number is derivable):

- A retrieved chunk is RELEVANT iff its resolved ``(roadmap_id,
  chunk_index)`` reference is in the case's ``expected`` set (binary
  relevance).
- Graded relevance (for nDCG@K): grade 2 = exact expected chunk, grade 1 =
  any chunk from a roadmap referenced in ``expected``, grade 0 = unrelated.
- Metrics are computed ONLY over KNOWLEDGE cases (``expected`` non-empty).
  NO-KNOWLEDGE cases (``expected`` empty) are evaluated separately: they
  pass when no retrieved chunk exceeds the strong-match threshold, i.e.
  the system does not fabricate a confident match where the roadmap corpus
  has no relevant information.
- ``retrieval_success_rate`` = Hit Rate@K over knowledge cases.
- IDCG@K is the case's ideal ranking: all expected chunks (grade 2, up to
  K) ranked first, then grade-1 chunks (other chunks of expected roadmaps)
  while slots remain, then grade 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel

from sharek_agents.agents.roadmap_rag.evaluation.dataset import EvaluationCase
from sharek_agents.agents.roadmap_rag.evaluation.embeddings import (
    deterministic_vector,
)
from sharek_agents.agents.roadmap_rag.evaluation.store import (
    InMemoryRoadmapStore,
)
from sharek_agents.agents.roadmap_rag.retrieval import (
    RealRoadmapRetriever,
    build_search_text,
)

STRONG_MATCH_THRESHOLD = 0.25
"""Max cosine similarity for a no-knowledge case to count as passed."""


def _discount(position: int) -> float:
    """log2-based rank discount (rank positions are 0-based)."""
    return math.log2(position + 2)


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """DCG@K over the retrieved relevance gains in rank order."""
    return sum(gain / _discount(position) for position, gain in enumerate(gains[:k]))


def ndcg_at_k(
    retrieved_gains: Sequence[float],
    ideal_gains: Sequence[float],
    k: int,
) -> float:
    """nDCG@K with the ideal ranking derived from expected + grade-1 chunks."""
    ideal = sorted(ideal_gains[:k], reverse=True)
    total = dcg_at_k(ideal, k)
    if total <= 0.0:
        return 0.0
    return dcg_at_k(retrieved_gains, k) / total


def _mean(values: Sequence[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def hit_rate_at_k(outcomes: Sequence[CaseOutcome]) -> float:
    """Fraction of knowledge cases with at least one relevant chunk in top-K."""
    return _mean(outcome.hit for outcome in outcomes)


def precision_at_k(outcomes: Sequence[CaseOutcome]) -> float:
    """Mean over knowledge cases of relevant_retrieved / limit."""
    return _mean(outcome.relevant_retrieved / outcome.limit for outcome in outcomes)


def recall_at_k(outcomes: Sequence[CaseOutcome]) -> float:
    """Mean over knowledge cases of relevant_retrieved / expected_count."""
    return _mean(outcome.recall for outcome in outcomes)


def mrr(outcomes: Sequence[CaseOutcome]) -> float:
    """Mean reciprocal rank of the first relevant chunk over knowledge cases."""
    return _mean(outcome.mrr for outcome in outcomes)


def ndcg_at_k_mean(outcomes: Sequence[CaseOutcome]) -> float:
    """Mean nDCG@K over knowledge cases."""
    return _mean(outcome.ndcg for outcome in outcomes)


class RefFound(BaseModel):
    """One retrieved chunk resolved to its ground-truth reference."""

    roadmap_id: int
    chunk_index: int
    cosine_similarity: float
    content: str


class CaseOutcome(BaseModel):
    """Per-case retrieval outcome with every metric component."""

    case_id: int
    skill: str
    query: str
    limit: int
    expected: list[tuple[int, int]]
    retrieved: list[RefFound]
    hit: bool
    relevant_retrieved: int
    precision: float
    recall: float
    mrr: float
    ndcg: float

    @property
    def is_knowledge_case(self) -> bool:
        return bool(self.expected)


class FailedCase(BaseModel):
    """Knowledge case that did not achieve a hit."""

    case_id: int
    skill: str
    query: str
    expected: list[tuple[int, int]]
    retrieved: list[RefFound]
    reason: str


class RefinementOutcome(BaseModel):
    """Query vs refined_query retrieval comparison (refinement case)."""

    case_id: int
    query: str
    refined_query: str
    initial_hit: bool
    refined_hit: bool
    refined_improves: bool


class EvaluationReport(BaseModel):
    """Aggregated, fully derived RAG evaluation report."""

    k: int
    total_cases: int
    knowledge_cases: int
    no_knowledge_cases: int
    hit_rate_at_k: float
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    retrieval_success_rate: float
    strong_match_threshold: float
    no_knowledge_passed: bool
    failed_cases: list[FailedCase]
    refinements: list[RefinementOutcome]
    outcomes: list[CaseOutcome]

    @classmethod
    def aggregate(
        cls,
        outcomes: Sequence[CaseOutcome],
        *,
        strong_match_threshold: float = STRONG_MATCH_THRESHOLD,
    ) -> EvaluationReport:
        if not outcomes:
            raise ValueError("cannot aggregate an empty evaluation")
        knowledge = [outcome for outcome in outcomes if outcome.is_knowledge_case]
        no_knowledge = [
            outcome for outcome in outcomes if not outcome.is_knowledge_case
        ]
        failed = [
            FailedCase(
                case_id=outcome.case_id,
                skill=outcome.skill,
                query=outcome.query,
                expected=outcome.expected,
                retrieved=outcome.retrieved,
                reason="no expected chunk in top-K",
            )
            for outcome in knowledge
            if not outcome.hit
        ]
        no_knowledge_passed = all(
            not outcome.retrieved
            or max(found.cosine_similarity for found in outcome.retrieved)
            < strong_match_threshold
            for outcome in no_knowledge
        )
        return cls(
            k=max(outcome.limit for outcome in outcomes),
            total_cases=len(outcomes),
            knowledge_cases=len(knowledge),
            no_knowledge_cases=len(no_knowledge),
            hit_rate_at_k=hit_rate_at_k(knowledge),
            precision_at_k=precision_at_k(knowledge),
            recall_at_k=recall_at_k(knowledge),
            mrr=mrr(knowledge),
            ndcg_at_k=ndcg_at_k_mean(knowledge),
            retrieval_success_rate=hit_rate_at_k(knowledge),
            strong_match_threshold=strong_match_threshold,
            no_knowledge_passed=no_knowledge_passed,
            failed_cases=failed,
            refinements=[],
            outcomes=list(outcomes),
        )


async def evaluate_cases(
    cases: Sequence[EvaluationCase],
    retriever: RealRoadmapRetriever,
    store: InMemoryRoadmapStore,
) -> list[CaseOutcome]:
    """Run every case through the REAL ``RoadmapRetriever`` contract.

    Retrieved ``RoadmapChunk`` objects (skill/topic/content only) are
    resolved back to ``(roadmap_id, chunk_index)`` references through the
    unique fixture chunk content. Cosine similarities come from a direct
    store search with the same deterministic search-text embedding, keeping
    the retriever's public ``build_search_text`` as the single source of
    the search-text construction.
    """
    ref_lookup = store.chunk_reference_lookup()
    outcomes: list[CaseOutcome] = []
    for case in cases:
        chunks = await retriever.search(
            skill=case.skill,
            query=case.query,
            current_level=case.current_level,
            target_level=case.target_level,
            gap_description=case.gap_description,
            limit=case.limit,
        )
        search_text = build_search_text(
            case.skill,
            case.query,
            case.current_level,
            case.target_level,
            case.gap_description,
        )
        ranked = await store.search_chunks(
            deterministic_vector(search_text), case.limit
        )
        similarities = {
            (item["roadmap_id"], item["chunk_index"]): item["cosine_similarity"]
            for item in ranked
        }
        expected_set = {ref for ref in case.expected}
        grade1_roadmaps = case.expected_roadmaps
        retrieved: list[RefFound] = []
        gains: list[float] = []
        for chunk in chunks:
            ref = ref_lookup.get(chunk.content)
            if ref is None:
                raise ValueError(
                    f"retrieved chunk content not in fixture corpus: "
                    f"{chunk.content[:60]}"
                )
            roadmap_id, chunk_index = ref
            retrieved.append(
                RefFound(
                    roadmap_id=roadmap_id,
                    chunk_index=chunk_index,
                    cosine_similarity=similarities.get(ref, 0.0),
                    content=chunk.content,
                )
            )
            gains.append(
                2.0
                if ref in expected_set
                else 1.0
                if roadmap_id in grade1_roadmaps
                else 0.0
            )
        relevant_retrieved = len(expected_set & retrieved_refs(retrieved))
        outcomes.append(
            CaseOutcome(
                case_id=case.id,
                skill=case.skill,
                query=case.query,
                limit=case.limit,
                expected=list(case.expected),
                retrieved=retrieved,
                hit=relevant_retrieved > 0,
                relevant_retrieved=relevant_retrieved,
                precision=relevant_retrieved / case.limit,
                recall=(
                    relevant_retrieved / len(case.expected)
                    if case.expected
                    else 0.0
                ),
                mrr=_mrr(retrieved, expected_set),
                ndcg=ndcg_at_k(gains, _ideal_gains(case, store), case.limit),
            )
        )
    return outcomes


def retrieved_refs(retrieved: Sequence[RefFound]) -> set[tuple[int, int]]:
    return {(found.roadmap_id, found.chunk_index) for found in retrieved}


def _mrr(retrieved: Sequence[RefFound], expected_set: set[tuple[int, int]]) -> float:
    for position, found in enumerate(retrieved):
        if (found.roadmap_id, found.chunk_index) in expected_set:
            return 1.0 / (position + 1)
    return 0.0


def _ideal_gains(case: EvaluationCase, store: InMemoryRoadmapStore) -> list[float]:
    """Ideal relevance gains: expected chunks (grade 2) then grade-1 chunks."""
    grade2 = len(case.expected)
    grade1 = sum(
        1
        for (roadmap_id, _chunk_index) in store.chunk_reference_lookup().values()
        if roadmap_id in case.expected_roadmaps
    )
    return [2.0] * grade2 + [1.0] * max(0, grade1 - grade2)
