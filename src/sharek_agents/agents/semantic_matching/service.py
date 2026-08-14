"""Semantic Matching service layer (Phases 1-4).

Phase 1 exposes matching-index operations over the PostgreSQL + pgvector
store through the repository layer only — never as an HTTP CRUD API.

Phase 2 wires the source-data abstraction: the service holds a
:class:`SourceDataProvider` (the read-only path to the authoritative main
database). Phase 9 implements the real provider
(``main_db.MainDbSourceDataProvider``) over the Main DB; its connection URL
and SQL queries are TODO placeholders in ``main_db.py`` until manually
completed.

Phase 3 wires embedding generation: ``embed_project`` / ``embed_contributor``
run the Phase 2 canonical representation through the Phase 3 embedding
service and store the validated vector in the matching index together with
the embedding metadata and the source freshness fields.

Phase 4 wires LAZY INDEXING + FRESHNESS CHECKING: ``ensure_project_indexed`` /
``ensure_contributor_indexed`` decide whether the local matching record is
current and only generate an embedding when the entity is missing locally or
stale:

    Entity id
        |
        v
    check local matching index
        |
        v
    not found -----> fetch source -> embed -> store
    found+current -> reuse stored vector (no embedding call)
    found+stale  -> fetch source -> embed -> replace record

The authoritative source data always decides currency; the stored
``source_version`` / ``source_updated_at`` come from the source, never from
the local clock.

Phase 8 implements the PRODUCTION MATCHING operation:
``match_projects_for_contributor`` answers "which indexed Projects match
this Contributor?" using pure cosine similarity over the matching index
(no skill filter, no reranker, no threshold). The Contributor's stored
embedding is refreshed through the Phase 4 lazy-indexing decision, and
Project vectors are read from the index as-is.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from sharek_agents.agents.semantic_matching.embeddings import (
    SemanticMatchingEmbeddingService,
    create_semantic_matching_embedding_service,
)
from sharek_agents.agents.semantic_matching.freshness import is_current
from sharek_agents.agents.semantic_matching.representation import (
    EMBEDDING_REPRESENTATION_VERSION,
)
from sharek_agents.agents.semantic_matching.schemas import (
    ContributorMatchRecord,
    ContributorSourceData,
    ProjectMatch,
    ProjectMatchRecord,
    ProjectSourceData,
    SemanticMatchResponse,
)
from sharek_agents.agents.semantic_matching.source_data import (
    EntityNotFoundError,
    SourceDataProvider,
    create_source_data_provider,
)
from sharek_agents.agents.semantic_matching.storage import (
    MatchingStorageError,
    SemanticMatchingStore,
    create_matching_store,
)


# ── Error hierarchy ───────────────────────────────────────────────────────────


class SemanticMatchingIndexingError(Exception):
    """The generated embedding could not be persisted to the matching index."""


class SemanticMatchingMatchError(Exception):
    """The matching query could not be completed against the matching index."""


# ── Service ───────────────────────────────────────────────────────────────────


class SemanticMatchingService:
    """Owns the matching index, the authoritative source data access, and
    embedding generation.

    The full Phase 3 -> Phase 4 flow:

        SourceData
            |
            v
        build_embedding_input()  (canonical representation)
            |
            v
        SemanticMatchingEmbeddingService
            |
            v
        validated vector
            |
            v
        PostgresSemanticMatchingStore (upsert_project / upsert_contributor)

    Phase 4 ``ensure_*_indexed`` wraps the flow with the lazy-index
    decision: check the local index first, fetch the authoritative source,
    and only regenerate the embedding when the local record is missing or
    stale.
    """

    def __init__(
        self,
        store: SemanticMatchingStore,
        source_data_provider: SourceDataProvider | None = None,
        embedding_service: SemanticMatchingEmbeddingService | None = None,
    ) -> None:
        self._store = store
        self._source_data_provider = create_source_data_provider(
            source_data_provider
        )
        self._embedding_service = (
            embedding_service or create_semantic_matching_embedding_service()
        )

    @property
    def source_data_provider(self) -> SourceDataProvider:
        """Read-only path to the authoritative main-database data."""
        return self._source_data_provider

    @property
    def embedding_service(self) -> SemanticMatchingEmbeddingService:
        """Phase 3 embedding adapter (canonical text -> validated vector)."""
        return self._embedding_service

    async def initialize(self) -> None:
        await self._store.initialize()

    async def upsert_project(self, record: ProjectMatchRecord) -> ProjectMatchRecord:
        return await self._store.upsert_project(record)

    async def upsert_contributor(
        self, record: ContributorMatchRecord
    ) -> ContributorMatchRecord:
        return await self._store.upsert_contributor(record)

    async def get_project(self, project_id: int) -> ProjectMatchRecord | None:
        return await self._store.get_project(project_id)

    async def get_contributor(self, contributor_id: int) -> ContributorMatchRecord | None:
        return await self._store.get_contributor(contributor_id)

    async def embed_project(self, data: ProjectSourceData) -> ProjectMatchRecord:
        """Generate and persist the embedding for one Project.

        Canonical text -> embedding vector -> matching-index record with
        embedding metadata (``embedding_model``, ``embedding_model_version``,
        ``embedding_schema_version`` = the Phase 2 representation version)
        and the source freshness fields carried by ``data``.

        The actual authoritative source metadata is supplied later by the
        Main DB integration; this method stores whatever version marker the
        caller provides.
        """
        vector = await self._embedding_service.embed(data)
        record = ProjectMatchRecord(
            project_id=data.project_id,
            skills=data.skills,
            evidence=data.evidence,
            source_version=data.source_version,
            source_updated_at=data.source_updated_at,
            embedding=vector,
            embedding_model=self._embedding_service.model,
            embedding_model_version=self._embedding_service.model_version,
            embedding_schema_version=EMBEDDING_REPRESENTATION_VERSION,
        )
        try:
            return await self._store.upsert_project(record)
        except MatchingStorageError as exc:
            raise SemanticMatchingIndexingError(
                f"Failed to persist the embedding for project_id "
                f"{data.project_id}: {exc}"
            ) from exc

    async def embed_contributor(
        self, data: ContributorSourceData
    ) -> ContributorMatchRecord:
        """Generate and persist the embedding for one Contributor.

        Mirror of :meth:`embed_project` for contributors.
        """
        vector = await self._embedding_service.embed(data)
        record = ContributorMatchRecord(
            contributor_id=data.contributor_id,
            skills=data.skills,
            evidence=data.evidence,
            source_version=data.source_version,
            source_updated_at=data.source_updated_at,
            embedding=vector,
            embedding_model=self._embedding_service.model,
            embedding_model_version=self._embedding_service.model_version,
            embedding_schema_version=EMBEDDING_REPRESENTATION_VERSION,
        )
        try:
            return await self._store.upsert_contributor(record)
        except MatchingStorageError as exc:
            raise SemanticMatchingIndexingError(
                f"Failed to persist the embedding for contributor_id "
                f"{data.contributor_id}: {exc}"
            ) from exc

    async def ensure_project_indexed(self, project_id: int) -> ProjectMatchRecord:
        """Lazily index a Project: return its current matching record.

        Decision (Phase 4):

        - not found in the local index -> fetch authoritative source data,
          generate the embedding, store the record;
        - found and current (freshness markers equal) -> reuse the stored
          vector, no embedding call;
        - found but stale (source markers differ) -> fetch authoritative
          source data, generate a new embedding, replace the record.

        The stored ``source_version`` / ``source_updated_at`` always come
        from the authoritative source data; the local clock is never used.

        Raises:
            EntityNotFoundError: The main database has no such Project.
            SourceDataProviderError: Authoritative source data is unavailable.
            InvalidFreshnessMetadataError: The source exposes no freshness
                marker, so currency cannot be verified.
            SemanticMatchingEmbeddingError: Embedding generation failed.
            SemanticMatchingIndexingError: The local index could not be read
                or written.
        """
        local = await self._read_project(project_id)
        source = await self._get_project_source(project_id)
        if local is not None and is_current(
            source.source_version,
            source.source_updated_at,
            local.source_version,
            local.source_updated_at,
        ):
            return local
        return await self.embed_project(source)

    async def ensure_contributor_indexed(
        self, contributor_id: int
    ) -> ContributorMatchRecord:
        """Lazily index a Contributor: return its current matching record.

        Mirror of :meth:`ensure_project_indexed` for contributors.
        """
        local = await self._read_contributor(contributor_id)
        source = await self._get_contributor_source(contributor_id)
        if local is not None and is_current(
            source.source_version,
            source.source_updated_at,
            local.source_version,
            local.source_updated_at,
        ):
            return local
        return await self.embed_contributor(source)

    async def match_projects_for_contributor(
        self, contributor_id: int, top_k: int = 10
    ) -> SemanticMatchResponse:
        """Rank matching Projects for a Contributor (Phase 8 production flow).

        The production matching decision is PURE COSINE SIMILARITY over the
        matching index:

        - the Contributor's embedding is obtained through the existing
          Phase 4 lazy-indexing decision (:meth:`ensure_contributor_indexed`),
          so missing/stale contributor data is handled there;
        - Project vectors are read from the index as-is; no Project
          embedding is regenerated;
        - no skill filter, no reranker, and no threshold is applied;
        - all indexed Projects are ranked by cosine similarity descending
          and truncated to ``top_k``.

        Raises:
            EntityNotFoundError: The main database has no such Contributor.
            SourceDataProviderError: Authoritative source data is unavailable.
            InvalidFreshnessMetadataError: The source exposes no freshness
                marker, so the Contributor's currency cannot be verified.
            SemanticMatchingEmbeddingError: Embedding generation failed.
            SemanticMatchingIndexingError: The Contributor could not be
                indexed.
            SemanticMatchingMatchError: The matching index could not be
                queried, or the Contributor has no stored embedding.
        """
        record = await self.ensure_contributor_indexed(contributor_id)
        if record.embedding is None:
            raise SemanticMatchingMatchError(
                f"Contributor {contributor_id} has no stored embedding; "
                "matching is not possible."
            )
        try:
            ranked = await self._store.find_projects_by_similarity(
                record.embedding, limit=top_k
            )
        except MatchingStorageError as exc:
            raise SemanticMatchingMatchError(
                f"Failed to query the matching index for contributor_id "
                f"{contributor_id}: {exc}"
            ) from exc
        matches = [
            ProjectMatch(
                project_id=item["project_id"],
                cosine_similarity=item["cosine_similarity"],
                rank=rank,
            )
            for rank, item in enumerate(ranked, start=1)
        ]
        return SemanticMatchResponse(
            contributor_id=contributor_id,
            top_k=top_k,
            matches=matches,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    async def _read_project(self, project_id: int) -> ProjectMatchRecord | None:
        try:
            return await self._store.get_project(project_id)
        except MatchingStorageError as exc:
            raise SemanticMatchingIndexingError(
                f"Failed to read the local matching index for project_id "
                f"{project_id}: {exc}"
            ) from exc

    async def _read_contributor(
        self, contributor_id: int
    ) -> ContributorMatchRecord | None:
        try:
            return await self._store.get_contributor(contributor_id)
        except MatchingStorageError as exc:
            raise SemanticMatchingIndexingError(
                f"Failed to read the local matching index for contributor_id "
                f"{contributor_id}: {exc}"
            ) from exc

    async def _get_project_source(self, project_id: int) -> ProjectSourceData:
        source = await self._source_data_provider.get_project(project_id)
        if source is None:
            raise EntityNotFoundError(
                f"The authoritative main database has no Project with "
                f"project_id {project_id}; nothing can be indexed."
            )
        return source

    async def _get_contributor_source(
        self, contributor_id: int
    ) -> ContributorSourceData:
        source = await self._source_data_provider.get_contributor(contributor_id)
        if source is None:
            raise EntityNotFoundError(
                f"The authoritative main database has no Contributor with "
                f"contributor_id {contributor_id}; nothing can be indexed."
            )
        return source


def create_semantic_matching_service(
    connection_provider: Callable[[], Awaitable[Any]] | None = None,
    source_data_provider: SourceDataProvider | None = None,
    embedding_service: SemanticMatchingEmbeddingService | None = None,
) -> SemanticMatchingService:
    """Build the Semantic Matching service over the matching index.

    ``source_data_provider`` defaults to the real Main DB provider
    (``main_db.MainDbSourceDataProvider``); its connection URL and SQL
    queries are TODO placeholders in ``main_db.py`` until manually
    completed.

    ``embedding_service`` defaults to the Phase 3 adapter backed by the
    repository's shared embedding configuration; it fails lazily when
    embedding is actually requested and no API key is configured.
    """
    return SemanticMatchingService(
        store=create_matching_store(connection_provider),
        source_data_provider=source_data_provider,
        embedding_service=embedding_service,
    )