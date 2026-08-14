from fastapi import HTTPException, status

from .schemas import RequirementInferenceInput, RequirementInferenceResult
from .service import (
    RequirementInferenceProviderError,
    RequirementInferenceProviderTimeout,
    infer_requirements,
)


async def analyze_requirement_inference(
    body: RequirementInferenceInput,
) -> RequirementInferenceResult:
    try:
        return await infer_requirements(body)
    except RequirementInferenceProviderTimeout as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Requirement inference provider timed out",
        ) from exc
    except RequirementInferenceProviderError as exc:
        # Deliberately opaque. The caller learns the attempt failed and is
        # retriable; it learns nothing about the provider, the model, or what
        # the model actually returned.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Requirement inference provider returned an invalid response",
        ) from exc
