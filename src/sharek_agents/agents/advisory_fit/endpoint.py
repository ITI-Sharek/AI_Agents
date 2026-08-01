from __future__ import annotations

from fastapi import HTTPException, status

from sharek_agents.agents.advisory_fit.schemas import (
    AdvisoryFitInput,
    AdvisoryFitResult,
)
from sharek_agents.agents.advisory_fit.service import (
    AdvisoryFitProviderError,
    AdvisoryFitProviderTimeout,
    generate_advisory_fit,
)
from sharek_agents.common.logging import get_logger


logger = get_logger(__name__)


async def analyze_advisory_fit(
    body: AdvisoryFitInput,
) -> AdvisoryFitResult:
    try:
        return await generate_advisory_fit(body)
    except AdvisoryFitProviderTimeout as exc:
        logger.warning("Advisory Fit provider timed out: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Advisory Fit provider timed out",
        ) from exc
    except AdvisoryFitProviderError as exc:
        logger.warning("Advisory Fit provider error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Advisory Fit provider returned an invalid response",
        ) from exc
