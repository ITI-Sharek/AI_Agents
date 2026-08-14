"""Semantic Matching endpoints (Phase 1 revision, Phase 8 wiring).

The public/internal feature API is ONE endpoint:

    POST /semantic-matching/match

with a body containing ONLY exactly one of ``contributor_id`` /
``project_id`` and ``top_k`` (see ``schemas.SemanticMatchRequest``).
The direction is determined by which id is present:
``contributor_id`` -> matching Projects, ``project_id`` -> matching
Contributors.

Phase 8 wires the Contributor -> Projects direction only, using the
production matching flow (pure cosine similarity over the matching index,
see ``service.SemanticMatchingService.match_projects_for_contributor``).
The Project -> Contributors direction is not implemented yet; such
requests fail with HTTP 501.

The matching index is an internal implementation detail accessed through
the service/repository layer. No storage CRUD endpoints are exposed.
"""

from __future__ import annotations

from fastapi import HTTPException, status

from sharek_agents.agents.semantic_matching.database import create_connection_provider
from sharek_agents.agents.semantic_matching.embeddings import (
    SemanticMatchingEmbeddingError,
)
from sharek_agents.agents.semantic_matching.freshness import (
    InvalidFreshnessMetadataError,
)
from sharek_agents.agents.semantic_matching.main_db import (
    MainDatabaseConfigurationError,
    MainDatabaseFreshnessError,
    MainDatabaseQueryError,
)
from sharek_agents.agents.semantic_matching.schemas import (
    SemanticMatchRequest,
    SemanticMatchResponse,
)
from sharek_agents.agents.semantic_matching.service import (
    SemanticMatchingIndexingError,
    SemanticMatchingMatchError,
    SemanticMatchingService,
    create_semantic_matching_service,
)
from sharek_agents.agents.semantic_matching.source_data import (
    EntityNotFoundError,
    SourceDataNotAvailableError,
)
from sharek_agents.agents.semantic_matching.storage import (
    MatchingStorageConfigurationError,
)
from sharek_agents.common.logging import get_logger
from sharek_agents.config import settings

logger = get_logger(__name__)

MATCH_ENDPOINT_PATH = "/semantic-matching/match"
"""The single matching endpoint path (registered in ``main.py``)."""


def _matching_service() -> SemanticMatchingService:
    """Build the production matching service from the configured settings.

    The matching index connection comes from
    ``SEMANTIC_MATCHING_DATABASE_URL``; the source data provider defaults to
    the real Main DB provider (``main_db.py``), whose connection URL and
    SQL queries are TODO placeholders until manually completed — surfaced
    as an explicit 503/502 at match time.
    """
    database_url = settings.semantic_matching_database_url
    if not database_url:
        raise MatchingStorageConfigurationError(
            "SEMANTIC_MATCHING_DATABASE_URL is not configured; the matching "
            "index connection cannot be created."
        )
    return create_semantic_matching_service(
        connection_provider=create_connection_provider(database_url)
    )


async def match_projects(body: SemanticMatchRequest) -> SemanticMatchResponse:
    """Run the production matching flow for one matching request."""
    if body.project_id is not None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                "Project -> Contributors matching is not implemented yet; "
                "only Contributor -> Projects matching is available."
            ),
        )
    try:
        return await _matching_service().match_projects_for_contributor(
            body.contributor_id, top_k=body.top_k
        )
    except EntityNotFoundError as exc:
        logger.warning("Match query for unknown contributor: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contributor {body.contributor_id} does not exist",
        ) from exc
    except MainDatabaseConfigurationError as exc:
        logger.error("Main DB not configured: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The authoritative Main DB is not configured",
        ) from exc
    except (MainDatabaseQueryError, MainDatabaseFreshnessError) as exc:
        logger.warning("Main DB read failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authoritative source data could not be read",
        ) from exc
    except SourceDataNotAvailableError as exc:
        logger.warning("Source data unavailable during match: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authoritative source data is not available",
        ) from exc
    except (
        InvalidFreshnessMetadataError,
        SemanticMatchingEmbeddingError,
        SemanticMatchingIndexingError,
        SemanticMatchingMatchError,
    ) as exc:
        logger.warning("Matching flow failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The matching index could not answer the query",
        ) from exc
    except MatchingStorageConfigurationError as exc:
        logger.error("Matching storage not configured: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Semantic matching storage is not configured",
        ) from exc