from __future__ import annotations

from fastapi import HTTPException, status

from sharek_agents.agents.gap_guidance.schemas import (
    GapGuidanceInput,
    GapGuidanceResult,
)
from sharek_agents.agents.gap_guidance.service import (
    GapGuidanceProviderError,
    GapGuidanceProviderTimeout,
    generate_gap_guidance,
)
from sharek_agents.common.logging import get_logger

logger = get_logger(__name__)


async def analyze_gap_guidance(
    body: GapGuidanceInput,
) -> GapGuidanceResult:
    try:
        return await generate_gap_guidance(body)
    except GapGuidanceProviderTimeout as exc:
        logger.warning("Gap Guidance provider timed out: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Gap Guidance provider timed out",
        ) from exc
    except GapGuidanceProviderError as exc:
        logger.warning("Gap Guidance provider error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gap Guidance provider returned an invalid response",
        ) from exc