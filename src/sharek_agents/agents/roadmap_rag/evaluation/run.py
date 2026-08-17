"""Deterministic Roadmap RAG evaluation runner.

Composes the evaluation end to end WITHOUT any LLM or credentials:

    fixtures -> RoadmapIngestionService (chunk_size=1)
             -> InMemoryRoadmapStore (deterministic embeddings)
             -> RealRoadmapRetriever (the production retrieval path)
             -> evaluation cases -> retrieval metrics -> report

Run with ``python -m sharek_agents.agents.roadmap_rag.evaluation.run``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from sharek_agents.agents.roadmap_rag.embeddings import RoadmapEmbeddingService
from sharek_agents.agents.roadmap_rag.evaluation.dataset import (
    EVALUATION_CASES,
    EvaluationCase,
    validate_dataset,
)
from sharek_agents.agents.roadmap_rag.evaluation.embeddings import (
    DeterministicEmbeddingService,
)
from sharek_agents.agents.roadmap_rag.evaluation.fixtures import ROADMAP_FIXTURES
from sharek_agents.agents.roadmap_rag.evaluation.metrics import (
    STRONG_MATCH_THRESHOLD,
    CaseOutcome,
    EvaluationReport,
    RefinementOutcome,
    evaluate_cases,
)
from sharek_agents.agents.roadmap_rag.evaluation.store import InMemoryRoadmapStore
from sharek_agents.agents.roadmap_rag.ingestion import RoadmapIngestionService
from sharek_agents.agents.roadmap_rag.retrieval import RealRoadmapRetriever


async def ingest_fixtures(
    store: InMemoryRoadmapStore,
    embedding_service: RoadmapEmbeddingService,
) -> None:
    """Ingest every fixture through the real ingestion service.

    ``chunk_size=1`` makes every ordered step exactly one chunk, so
    ``chunk_index`` equals the step order and the dataset ground truth
    ``(roadmap_id, chunk_index)`` references are exact.
    """
    service = RoadmapIngestionService(
        store=store, embedding_service=embedding_service
    )
    for fixture in ROADMAP_FIXTURES:
        await service.ingest(
            fixture.title, "\n".join(fixture.steps), chunk_size=1
        )


async def build_deterministic_environment(
    store: InMemoryRoadmapStore | None = None,
    embedding: DeterministicEmbeddingService | None = None,
) -> tuple[InMemoryRoadmapStore, RoadmapEmbeddingService, DeterministicEmbeddingService]:
    """Build store + embedding wiring and ingest the fixtures once."""
    if store is None:
        store = InMemoryRoadmapStore()
        await store.initialize()
    if embedding is None:
        embedding = DeterministicEmbeddingService()
    roadmap_embeddings = RoadmapEmbeddingService(service=embedding)
    await ingest_fixtures(store, roadmap_embeddings)
    return store, roadmap_embeddings, embedding


async def run_evaluation(
    cases: Sequence[EvaluationCase] | None = None,
    *,
    strong_match_threshold: float = STRONG_MATCH_THRESHOLD,
) -> EvaluationReport:
    """Run the full deterministic evaluation and aggregate the report."""
    if cases is None:
        cases = EVALUATION_CASES
    validate_dataset()
    store, roadmap_embeddings, _ = await build_deterministic_environment()
    retriever = RealRoadmapRetriever(
        store=store, embedding_service=roadmap_embeddings
    )
    outcomes = await evaluate_cases(cases, retriever, store)
    report = EvaluationReport.aggregate(
        outcomes, strong_match_threshold=strong_match_threshold
    )
    refinements = await _refinement_outcomes(cases, retriever, store, outcomes)
    return report.model_copy(update={"refinements": refinements})


async def _refinement_outcomes(
    cases: Sequence[EvaluationCase],
    retriever: RealRoadmapRetriever,
    store: InMemoryRoadmapStore,
    outcomes: Sequence[CaseOutcome],
) -> list[RefinementOutcome]:
    """Evaluate refined queries against the SAME retrieval contract."""
    ref_lookup = store.chunk_reference_lookup()
    outcome_by_id = {outcome.case_id: outcome for outcome in outcomes}
    results: list[RefinementOutcome] = []
    for case in cases:
        if not case.refined_query:
            continue
        initial = outcome_by_id[case.id]
        chunks = await retriever.search(
            skill=case.skill,
            query=case.refined_query,
            current_level=case.current_level,
            target_level=case.target_level,
            gap_description=case.gap_description,
            limit=case.limit,
        )
        retrieved = {
            ref_lookup[chunk.content] for chunk in chunks if chunk.content in ref_lookup
        }
        refined_hit = bool(retrieved & {ref for ref in case.expected})
        results.append(
            RefinementOutcome(
                case_id=case.id,
                query=case.query,
                refined_query=case.refined_query,
                initial_hit=initial.hit,
                refined_hit=refined_hit,
                refined_improves=refined_hit and not initial.hit,
            )
        )
    return results


def _pct(value: float) -> str:
    return f"{value:.2f} ({value * 100:.0f}%)"


def format_report(report: EvaluationReport) -> str:
    """Render the evaluation report as a readable text document."""
    lines = [
        "Roadmap RAG Evaluation",
        "=====================",
        "",
        "Setup",
        "-----",
        (
            f"- Roadmaps: {len(ROADMAP_FIXTURES)} fixtures (FastAPI, "
            "PostgreSQL, Software Architecture, Testing, Docker)"
        ),
        (
            "- Ingestion: RoadmapIngestionService with chunk_size=1, so each "
            "ordered step is exactly one chunk"
        ),
        (
            "- Embeddings: deterministic 2048-d bag-of-words (crc32 unigrams "
            "+ bigrams), reproducible across runs"
        ),
        (
            "- Retriever: RealRoadmapRetriever over an in-memory RoadmapStore "
            "(real search path, one query embedding per search)"
        ),
        (
            "- Relevance: retrieved chunk relevant iff (roadmap_id, "
            "chunk_index) is in the case's expected set"
        ),
        f"- K: {report.k}",
        "",
        f"Metrics over {report.knowledge_cases} knowledge cases (K={report.k})",
        "---------------------------------------------------------",
        (
            f"- Hit Rate@K:                 {_pct(report.hit_rate_at_k)} of "
            "cases with >= 1 relevant chunk in top-K"
        ),
        (
            f"- Precision@K:                {_pct(report.precision_at_k)} "
            "(mean relevant_retrieved / limit)"
        ),
        (
            f"- Recall@K:                   {_pct(report.recall_at_k)} (mean "
            "relevant_retrieved / expected)"
        ),
        (
            f"- MRR:                        {_pct(report.mrr)} (mean "
            "reciprocal rank of first relevant chunk)"
        ),
        (
            f"- nDCG@K:                     {_pct(report.ndcg_at_k)} (graded "
            "2=exact, 1=same roadmap, 0=unrelated)"
        ),
        (
            f"- Retrieval success rate:     "
            f"{_pct(report.retrieval_success_rate)} "
            "(= Hit Rate@K over knowledge cases)"
        ),
        "",
        (
            f"No-knowledge cases: {report.no_knowledge_cases} (empty expected "
            f"sets), passed = {report.no_knowledge_passed}"
        ),
        (
            f"- Pass criterion: no retrieved chunk exceeds the strong-match "
            f"threshold of {report.strong_match_threshold}"
        ),
        "",
    ]
    if report.failed_cases:
        lines.append("Failed cases (knowledge cases without a hit):")
        lines.append("-" * 47)
        for failed in report.failed_cases:
            expected = ", ".join(
                f"({roadmap_id}:{chunk_index})"
                for roadmap_id, chunk_index in failed.expected
            )
            retrieved = ", ".join(
                f"({found.roadmap_id}:{found.chunk_index}, "
                f"sim={found.cosine_similarity:.3f})"
                for found in failed.retrieved
            )
            lines.append(
                f"- case {failed.case_id} [{failed.skill}]: "
                f"expected {expected}, retrieved {retrieved} -- "
                f"{failed.reason}"
            )
        lines.append("")
    if report.refinements:
        lines.append("Refinement cases (focused re-query evaluation):")
        lines.append("-" * 55)
        for outcome in report.refinements:
            verdict = (
                "refined query retrieves the expected chunk"
                if outcome.refined_improves
                else "no improvement"
            )
            lines.append(
                f"- case {outcome.case_id}: initial hit={outcome.initial_hit}, "
                f"refined hit={outcome.refined_hit} -- {verdict}"
            )
        lines.append("")
    lines.append("Derived values: every metric above is computed directly from")
    lines.append("the retrieval results by evaluation/metrics.py; nothing is")
    lines.append("hardcoded. No LLM and no credentials were used.")
    return "\n".join(lines)


async def _run_main() -> None:
    report = await run_evaluation()
    print(format_report(report))


if __name__ == "__main__":
    asyncio.run(_run_main())
