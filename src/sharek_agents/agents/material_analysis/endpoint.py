from fastapi import HTTPException, status

from sharek_agents.agents.material_analysis.schemas import (
    MaterialAnalysisInput,
    MaterialAnalysisResult,
)
from sharek_agents.agents.material_analysis.service import (
    MaterialAnalysisInputError,
    MaterialAnalysisProviderError,
    MaterialAnalysisProviderTimeout,
    generate_material_analysis,
)


async def analyze_materials(body: MaterialAnalysisInput) -> MaterialAnalysisResult:
    try:
        return await generate_material_analysis(body)
    except MaterialAnalysisInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Selected Materials could not be analyzed",
        ) from exc
    except MaterialAnalysisProviderTimeout as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Material analysis provider timed out",
        ) from exc
    except MaterialAnalysisProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Material analysis provider returned an invalid response",
        ) from exc
