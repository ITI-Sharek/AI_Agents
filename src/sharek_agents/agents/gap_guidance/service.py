from __future__ import annotations

import logging

from sharek_agents.agents.gap_guidance.agent import (
    GapGuidanceAgent,
    GapGuidanceAgentConfig,
    GapGuidanceProviderError,
    GapGuidanceProviderTimeout,
)
from sharek_agents.agents.gap_guidance.schemas import (
    GapGuidanceInput,
    GapGuidanceResult,
)
from sharek_agents.agents.gap_guidance.tools import SearchRoadmapTool
from sharek_agents.agents.roadmap_rag.embeddings import RoadmapEmbeddingError
from sharek_agents.agents.roadmap_rag.ingestion import RoadmapIngestionError
from sharek_agents.agents.roadmap_rag.retrieval import (
    create_real_roadmap_retriever,
)
from sharek_agents.agents.roadmap_rag.storage import (
    RoadmapStorageError,
    RoadmapStore,
    create_roadmap_store,
)
from sharek_agents.agents.roadmap_rag.synchronization import (
    RoadmapSynchronizationService,
)
from sharek_agents.agents.semantic_matching.database import (
    create_connection_provider,
)
from sharek_agents.config import settings

logger = logging.getLogger(__name__)


def _production_roadmap_store() -> RoadmapStore:
    """Build the production roadmap store (shared Semantic Matching DB wiring)."""
    database_url = settings.semantic_matching_database_url
    connection_provider = (
        create_connection_provider(database_url) if database_url else None
    )
    return create_roadmap_store(connection_provider)


async def generate_gap_guidance(
    input_data: GapGuidanceInput,
) -> GapGuidanceResult:
    logger.info(
        "Starting Gap Guidance analysis: %d assessments",
        len(input_data.advisory_fit_result.assessments),
    )

    # Pre-retrieval synchronization: roadmaps inserted directly into the
    # ``roadmaps`` table (manual SQL inserts) are detected and processed
    # through ``RoadmapIngestionService`` BEFORE the retriever answers any
    # search. The Agent and the ``search_roadmap`` tool stay unaware of
    # this step; when the knowledge base cannot be synchronized the
    # request fails through the existing provider error mapping instead of
    # answering over an unprocessed knowledge base.
    store = _production_roadmap_store()
    try:
        await RoadmapSynchronizationService(store=store).synchronize()
    except (RoadmapStorageError, RoadmapIngestionError, RoadmapEmbeddingError) as exc:
        logger.error("Roadmap synchronization failed before retrieval: %s", exc)
        raise GapGuidanceProviderError(
            "Roadmap knowledge synchronization failed before retrieval"
        ) from exc

    # Production retrieval backend: the real Roadmap RAG retriever (the
    # Agent/``search_roadmap`` contract stays unchanged; the mock backend
    # remains available for tests through dependency injection).
    retriever = create_real_roadmap_retriever(store=store)
    agent = GapGuidanceAgent(
        config=GapGuidanceAgentConfig(
            tools=[SearchRoadmapTool(retriever=retriever)],
        )
    )
    return await agent.run(
        input_data.advisory_fit_result, answer=input_data.answer
    )


__all__ = [
    "GapGuidanceProviderError",
    "GapGuidanceProviderTimeout",
    "generate_gap_guidance",
]
