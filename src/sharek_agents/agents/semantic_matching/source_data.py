"""Semantic Matching source data abstraction (Phase 2) + Main DB wiring (Phase 9).

The main business database (owned by the backend/NestJS side) is the SOURCE
OF TRUTH for Project and Contributor data. Phase 2 defined the
abstraction/contract that the Semantic Matching service uses to receive
authoritative data:

    Semantic Matching Service
            |
            v
      Source Data Provider
            |
            v
      [Main DB]
            |
            v
    Authoritative entity data

Phase 9 implements the real provider (``main_db.MainDbSourceDataProvider``)
around this contract: it reads the Main DB through a DB connection/query
abstraction (never HTTP) and returns the authoritative entity data with the
Main DB ``updated_at`` as the freshness marker. The provider's connection
URL (``MAIN_DATABASE_URL``) and SQL queries are TODO placeholders in
``main_db.py`` that require manual completion; until then the provider
raises a clear ``MainDatabaseConfigurationError`` at first use.
"""

from __future__ import annotations

from typing import Protocol

from sharek_agents.agents.semantic_matching.schemas import (
    ContributorSourceData,
    ProjectSourceData,
)


# ── Error hierarchy ───────────────────────────────────────────────────────────


class SourceDataProviderError(Exception):
    """Base error for source data provider failures."""


class SourceDataNotAvailableError(SourceDataProviderError):
    """Raised when authoritative source data cannot be read."""


class EntityNotFoundError(SourceDataProviderError):
    """The authoritative main database has no record for the requested entity.

    Raised when the source data provider returns ``None`` for a requested
    Project/Contributor id: the entity does not exist in the main database,
    so nothing can be indexed.
    """


# ── Protocol ──────────────────────────────────────────────────────────────────


class SourceDataProvider(Protocol):
    """Reads authoritative Project/Contributor matching data.

    Read-only access to the main business database. Implementations are
    supplied by the deployment/backend wiring in a later phase.
    """

    async def get_project(self, project_id: int) -> ProjectSourceData | None:
        """Return the authoritative Project data, or None if unknown."""

    async def get_contributor(
        self, contributor_id: int
    ) -> ContributorSourceData | None:
        """Return the authoritative Contributor data, or None if unknown."""


# ── Factory ───────────────────────────────────────────────────────────────────


def create_source_data_provider(
    provider: SourceDataProvider | None = None,
) -> SourceDataProvider:
    """Build the source data provider.

    Defaults to the real Main DB provider (``main_db.py``): the Main DB is
    the authoritative source of truth. Its connection URL (``MAIN_DATABASE_URL``)
    and SQL queries are TODO placeholders in ``main_db.py`` that require
    manual completion; until then reads raise
    ``MainDatabaseConfigurationError``. Pass ``provider`` to override, e.g.
    with deployment wiring that already owns a Main DB connection.
    """
    if provider is not None:
        return provider
    from sharek_agents.agents.semantic_matching.main_db import (  # noqa: PLC0415 (local import breaks the main_db -> source_data cycle)
        create_main_db_source_data_provider,
    )

    return create_main_db_source_data_provider()