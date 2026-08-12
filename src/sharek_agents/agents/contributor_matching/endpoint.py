from __future__ import annotations

from fastapi import HTTPException, status

from sharek_agents.agents.contributor_matching.schemas import (
    ContributorMatchingInput,
    ContributorMatchingResult,
)
from sharek_agents.agents.contributor_matching.service import (
    ContributorMatchingProviderError,
    ContributorMatchingProviderTimeout,
    generate_contributor_matching,
)
from sharek_agents.common.logging import get_logger


logger = get_logger(__name__)


async def analyze_contributor_matching(
    body: ContributorMatchingInput,
) -> ContributorMatchingResult:
    try:
        return await generate_contributor_matching(body)
    except ContributorMatchingProviderTimeout as exc:
        logger.warning("Contributor matching provider timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Contributor matching provider timed out",
        ) from exc
    except ContributorMatchingProviderError as exc:
        logger.warning("Contributor matching provider returned invalid output")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Contributor matching provider returned an invalid response",
        ) from exc
