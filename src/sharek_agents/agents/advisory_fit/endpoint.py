from fastapi import HTTPException, status

from .schemas import AdvisoryFitInput, AdvisoryFitResult
from .service import (
    AdvisoryFitProviderError,
    AdvisoryFitProviderTimeout,
    generate_advisory_fit,
)


async def analyze_advisory_fit(body: AdvisoryFitInput) -> AdvisoryFitResult:
    try:
        return await generate_advisory_fit(body)
    except AdvisoryFitProviderTimeout as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Advisory Fit provider timed out",
        ) from exc
    except AdvisoryFitProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Advisory Fit provider returned an invalid response",
        ) from exc
