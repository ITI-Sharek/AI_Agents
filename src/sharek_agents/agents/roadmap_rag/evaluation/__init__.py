"""Deterministic Roadmap RAG retrieval evaluation.

Pure offline evaluation: fixture roadmaps ingested through the real
ingestion service (chunk_size=1), deterministic embeddings, an in-memory
store, the real ``RealRoadmapRetriever`` search path, and a ground-truth
dataset with documented RAG metrics. No LLM, no credentials.
"""

from sharek_agents.agents.roadmap_rag.evaluation.dataset import (
    EVALUATION_CASES,
    EvaluationCase,
    validate_dataset,
)
from sharek_agents.agents.roadmap_rag.evaluation.embeddings import (
    DeterministicEmbeddingService,
    deterministic_vector,
)
from sharek_agents.agents.roadmap_rag.evaluation.fixtures import (
    ROADMAP_FIXTURES,
    RoadmapFixture,
)
from sharek_agents.agents.roadmap_rag.evaluation.metrics import (
    STRONG_MATCH_THRESHOLD,
    CaseOutcome,
    EvaluationReport,
    FailedCase,
    RefinementOutcome,
    evaluate_cases,
)
from sharek_agents.agents.roadmap_rag.evaluation.run import (
    build_deterministic_environment,
    format_report,
    ingest_fixtures,
    run_evaluation,
)
from sharek_agents.agents.roadmap_rag.evaluation.store import InMemoryRoadmapStore

__all__ = [
    "EVALUATION_CASES",
    "ROADMAP_FIXTURES",
    "STRONG_MATCH_THRESHOLD",
    "CaseOutcome",
    "DeterministicEmbeddingService",
    "EvaluationCase",
    "EvaluationReport",
    "FailedCase",
    "InMemoryRoadmapStore",
    "RefinementOutcome",
    "RoadmapFixture",
    "build_deterministic_environment",
    "deterministic_vector",
    "evaluate_cases",
    "format_report",
    "ingest_fixtures",
    "run_evaluation",
    "validate_dataset",
]
