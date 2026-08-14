"""Semantic Matching Main DB source integration (Phase 9).

The Main DB (owned by the backend/NestJS side) is the SOURCE OF TRUTH for
Project and Contributor data. This module implements the real
``SourceDataProvider`` around the existing Phase 2 abstraction:

    Semantic Matching Service
            |
            v
      SourceDataProvider (SourceDataProvider protocol)
            |
            v
      MainDbSourceDataProvider (this module)
            |
            v
      Main DB (asyncpg connection/query abstraction, NOT HTTP)
            |
            v
    Authoritative entity data (id, skills, levels, updated_at)

Rules honored here:

- The Main DB is accessed ONLY through a DB connection/query abstraction
  (an injected asyncpg-style connection); no HTTP/API requests are used.
- The Main DB is the source of truth; the local Semantic Matching index
  timestamp is NEVER used as the authoritative freshness value. The
  provider returns the Main DB ``updated_at`` as ``source_updated_at``.
- The Main DB and the Semantic Matching index are TWO DIFFERENT databases.
  The Main DB URL is separate from ``SEMANTIC_MATCHING_DATABASE_URL`` and
  never replaces it.

MANUAL COMPLETION REQUIRED (intentionally left unfinished):

1. ``MAIN_DATABASE_URL`` — the real Main DB connection target.
2. The ``GET_*_QUERY`` SQL placeholders — the real Main DB queries.

Neither is guessed here. Until both are completed the provider raises a
clear ``MainDatabaseConfigurationError`` at first use; the SQL placeholders
document the exact result columns the provider maps by name.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Awaitable, Callable

from pydantic import ValidationError

from sharek_agents.agents.semantic_matching.schemas import (
    ContributorSourceData,
    ProjectSourceData,
    SkillItem,
)
from sharek_agents.agents.semantic_matching.source_data import (
    SourceDataProvider,
    SourceDataProviderError,
)


# ── Configuration placeholder (manual completion) ─────────────────────────────


MAIN_DATABASE_URL = "<FILL_IN_MAIN_DATABASE_URL>"
"""Main DB connection target (the authoritative source of truth).

MANUAL COMPLETION REQUIRED: fill in the real Main DB URL before the
provider can connect. The value is intentionally NOT guessed here.

The Main DB is a DIFFERENT database from the Semantic Matching index: it is
the authoritative source, while ``SEMANTIC_MATCHING_DATABASE_URL`` points at
the local matching/vector storage. This setting never replaces
``SEMANTIC_MATCHING_DATABASE_URL``.
"""


# ── SQL query placeholders (manual completion) ────────────────────────────────
#
# MANUAL COMPLETION REQUIRED: replace each placeholder with the real Main DB
# query. The provider maps result rows BY COLUMN NAME, so the real query MUST
# return the columns documented below (exact names). Each query is executed
# with one positional parameter: $1 = the entity id.


GET_CONTRIBUTOR_QUERY = """
-- TODO: INSERT REAL MAIN DB QUERY HERE (manual completion).
--
-- Required result columns (mapped by name):
--   contributor_id  integer       the Contributor id
--   skills          text[]        skill names
--   skill_levels    text[]        level per skill, same order as skills
--   updated_at      timestamptz   authoritative last-update timestamp
--
-- Executed with $1 = contributor_id.
"""

GET_CONTRIBUTOR_FRESHNESS_QUERY = """
-- TODO: INSERT REAL MAIN DB QUERY HERE (manual completion).
--
-- Required result columns (mapped by name):
--   contributor_id  integer       the Contributor id
--   updated_at      timestamptz   authoritative last-update timestamp
--
-- Executed with $1 = contributor_id.
-- Cheap freshness-only variant, used to verify currency before deciding
-- whether the full data must be fetched again.
"""

GET_PROJECT_QUERY = """
-- TODO: INSERT REAL MAIN DB QUERY HERE (manual completion).
--
-- Required result columns (mapped by name):
--   project_id      integer       the Project id
--   skills          text[]        skill names
--   skill_levels    text[]        level per skill, same order as skills
--   updated_at      timestamptz   authoritative last-update timestamp
--
-- Executed with $1 = project_id.
"""

GET_PROJECT_FRESHNESS_QUERY = """
-- TODO: INSERT REAL MAIN DB QUERY HERE (manual completion).
--
-- Required result columns (mapped by name):
--   project_id      integer       the Project id
--   updated_at      timestamptz   authoritative last-update timestamp
--
-- Executed with $1 = project_id.
-- Cheap freshness-only variant, used to verify currency before deciding
-- whether the full data must be fetched again.
"""


# ── Error hierarchy ───────────────────────────────────────────────────────────


class MainDatabaseError(SourceDataProviderError):
    """Base error for Main DB source data access."""


class MainDatabaseConfigurationError(MainDatabaseError):
    """Main DB connection/query configuration is missing or invalid.

    Raised when the ``MAIN_DATABASE_URL`` placeholder is still unfilled,
    the asyncpg driver is not installed, or the provider was built without
    a usable connection target.
    """


class MainDatabaseQueryError(MainDatabaseError):
    """The Main DB connection or query failed, or its row cannot be mapped.

    Covers connection failures, query execution failures, and rows that
    violate the documented result-column contract (missing columns,
    misaligned skills/levels, unknown skill levels).
    """


class MainDatabaseFreshnessError(MainDatabaseError):
    """The Main DB returned no authoritative ``updated_at`` for the entity.

    The local record can never be confirmed current without the
    authoritative freshness marker; the caller must surface the error
    instead of silently reusing the stored vector.
    """


# ── Connection wiring ─────────────────────────────────────────────────────────


def create_main_db_connection_provider(
    database_url: str | None = None,
) -> Callable[[], Awaitable[Any]]:
    """Build the Main DB connection provider (lazy asyncpg pool).

    The pool is created lazily on first use and reused (the provider asks
    for a connection per operation). Unlike the Semantic Matching index
    (``database.py``), the Main DB is a regular business database, so NO
    pgvector codec is registered here.

    Configuration safety: the Main DB URL is SEPARATE from
    ``SEMANTIC_MATCHING_DATABASE_URL``. ``database_url`` defaults to the
    ``MAIN_DATABASE_URL`` placeholder; until that placeholder is completed
    (manually or by deployment wiring passing ``database_url``), every
    connection attempt raises ``MainDatabaseConfigurationError``.

    asyncpg is imported lazily so importing this module never requires the
    driver; a clear error is raised only when a connection is actually
    requested without it installed.
    """

    url = database_url or MAIN_DATABASE_URL
    state: dict[str, Any] = {"pool": None}

    async def _connection() -> Any:
        if state["pool"] is not None:
            return state["pool"]
        if not url or url.startswith("<") or "FILL_IN" in url.upper():
            raise MainDatabaseConfigurationError(
                "Main DB connection is not configured. Fill in "
                "MAIN_DATABASE_URL in main_db.py (or pass database_url to "
                "create_main_db_connection_provider). The Main DB is NOT "
                "the Semantic Matching index "
                "(SEMANTIC_MATCHING_DATABASE_URL)."
            )
        try:
            import asyncpg  # noqa: PLC0415 (lazy: driver-agnostic design)
        except ImportError as exc:  # pragma: no cover - environment check
            raise MainDatabaseConfigurationError(
                "asyncpg is not installed; it is required to connect to the "
                "Main DB. Install asyncpg in the deployment environment."
            ) from exc
        try:
            state["pool"] = await asyncpg.create_pool(url)
        except Exception as exc:
            raise MainDatabaseQueryError(
                f"Could not connect to the Main DB: {exc}"
            ) from exc
        return state["pool"]

    async def provider() -> Any:
        return await _connection()

    return provider


# ── Row mapping ───────────────────────────────────────────────────────────────

_SKILL_LEVELS = ("beginner", "intermediate", "advanced")


def _skill_items(skills: Any, levels: Any) -> list[SkillItem]:
    """Map the Main DB ``skills`` / ``skill_levels`` columns to SkillItems.

    Both columns must align by index: the i-th level describes the i-th
    skill. NULL columns map to an empty list. Levels must be one of
    ``beginner`` / ``intermediate`` / ``advanced``; anything else is a
    contract violation and fails loudly.
    """
    skill_list = list(skills or [])
    level_list = list(levels or [])
    if len(skill_list) != len(level_list):
        raise MainDatabaseQueryError(
            f"Main DB returned {len(skill_list)} skills but "
            f"{len(level_list)} levels; the skills/skill_levels columns "
            "must align by index."
        )
    items: list[SkillItem] = []
    for skill, level in zip(skill_list, level_list):
        normalized = str(level).strip().lower()
        if normalized not in _SKILL_LEVELS:
            raise MainDatabaseQueryError(
                f"Main DB returned unknown skill level '{level}' for skill "
                f"'{skill}'; expected one of "
                f"beginner/intermediate/advanced."
            )
        items.append(SkillItem(skill=str(skill), level=normalized))
    return items


def _authoritative_updated_at(row: Any) -> datetime:
    """Return the Main DB ``updated_at`` column as the authoritative marker.

    A missing (NULL) marker means the local record can never be confirmed
    current; the provider raises instead of silently assuming currency.
    """
    updated_at = row["updated_at"]
    if updated_at is None:
        raise MainDatabaseFreshnessError(
            "The Main DB returned no updated_at for the entity; its "
            "freshness cannot be verified, so the local matching record "
            "cannot be confirmed current."
        )
    return updated_at


# ── Provider ──────────────────────────────────────────────────────────────────


class MainDbSourceDataProvider:
    """The real ``SourceDataProvider`` backed by the Main DB (source of truth).

    Reads authoritative entity data through an injected asyncpg-style
    connection (``fetchrow`` duck-type) using the SQL queries in this
    module (or queries supplied by the deployment wiring). The Main DB's
    authoritative ``updated_at`` is exposed as ``source_updated_at``; the
    local Semantic Matching index timestamp is never used as the freshness
    value.

    Contract:

    - ``get_contributor`` / ``get_project`` return the full authoritative
      data (id, skills, levels, ``source_updated_at``), or ``None`` when
      the Main DB has no such entity (the service turns that into
      ``EntityNotFoundError``).
    - ``get_contributor_freshness`` / ``get_project_freshness`` return only
      the authoritative ``updated_at``, or ``None`` when the entity does
      not exist (cheap currency check).

    Errors are distinguished:

    - unknown entity -> ``None`` (service raises ``EntityNotFoundError``);
    - connection/query failure or unmappable row ->
      :class:`MainDatabaseQueryError`;
    - unfilled/invalid Main DB configuration ->
      :class:`MainDatabaseConfigurationError`;
    - missing authoritative ``updated_at`` ->
      :class:`MainDatabaseFreshnessError`.
    """

    def __init__(
        self,
        connection_provider: Callable[[], Awaitable[Any]] | None = None,
        queries: dict[str, str] | None = None,
    ) -> None:
        self._connection_provider = (
            connection_provider or create_main_db_connection_provider()
        )
        self._queries = queries or {
            "get_contributor": GET_CONTRIBUTOR_QUERY,
            "get_contributor_freshness": GET_CONTRIBUTOR_FRESHNESS_QUERY,
            "get_project": GET_PROJECT_QUERY,
            "get_project_freshness": GET_PROJECT_FRESHNESS_QUERY,
        }

    async def get_contributor(
        self, contributor_id: int
    ) -> ContributorSourceData | None:
        """Return the authoritative Contributor data, or None if unknown."""
        row = await self._fetch_row("get_contributor", contributor_id)
        if row is None:
            return None
        return self._contributor_source(row)

    async def get_project(self, project_id: int) -> ProjectSourceData | None:
        """Return the authoritative Project data, or None if unknown."""
        row = await self._fetch_row("get_project", project_id)
        if row is None:
            return None
        return self._project_source(row)

    async def get_contributor_freshness(
        self, contributor_id: int
    ) -> datetime | None:
        """Return the authoritative Contributor ``updated_at``, or None."""
        return await self._freshness("get_contributor_freshness", contributor_id)

    async def get_project_freshness(self, project_id: int) -> datetime | None:
        """Return the authoritative Project ``updated_at``, or None."""
        return await self._freshness("get_project_freshness", project_id)

    # ── Private helpers ────────────────────────────────────────────────────────

    async def _fetch_row(self, query_key: str, entity_id: int) -> Any:
        try:
            connection = await self._connection_provider()
        except (MainDatabaseConfigurationError, MainDatabaseQueryError):
            raise
        except Exception as exc:
            raise MainDatabaseQueryError(
                f"Could not obtain a Main DB connection: {exc}"
            ) from exc
        try:
            return await connection.fetchrow(self._queries[query_key], entity_id)
        except Exception as exc:
            raise MainDatabaseQueryError(
                f"The Main DB query '{query_key}' failed: {exc}"
            ) from exc

    async def _freshness(self, query_key: str, entity_id: int) -> datetime | None:
        row = await self._fetch_row(query_key, entity_id)
        if row is None:
            return None
        return _authoritative_updated_at(row)

    def _contributor_source(self, row: Any) -> ContributorSourceData:
        try:
            return ContributorSourceData(
                contributor_id=row["contributor_id"],
                skills=_skill_items(row["skills"], row["skill_levels"]),
                evidence=[],
                source_version=None,
                source_updated_at=_authoritative_updated_at(row),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MainDatabaseQueryError(
                f"Main DB contributor row cannot be mapped to the "
                f"source-data contract: {exc}"
            ) from exc

    def _project_source(self, row: Any) -> ProjectSourceData:
        try:
            return ProjectSourceData(
                project_id=row["project_id"],
                skills=_skill_items(row["skills"], row["skill_levels"]),
                evidence=[],
                source_version=None,
                source_updated_at=_authoritative_updated_at(row),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MainDatabaseQueryError(
                f"Main DB project row cannot be mapped to the source-data "
                f"contract: {exc}"
            ) from exc


# ── Factory ───────────────────────────────────────────────────────────────────


def create_main_db_source_data_provider(
    connection_provider: Callable[[], Awaitable[Any]] | None = None,
    queries: dict[str, str] | None = None,
) -> MainDbSourceDataProvider:
    """Build the Main DB source data provider (the real SourceDataProvider).

    ``connection_provider`` / ``queries`` may be supplied by the deployment
    wiring; by default the provider uses the ``MAIN_DATABASE_URL`` and SQL
    placeholders in this module, which raise a clear configuration error
    until manually completed.
    """
    return MainDbSourceDataProvider(
        connection_provider=connection_provider,
        queries=queries,
    )