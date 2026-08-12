import os
import secrets

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from dotenv import load_dotenv

load_dotenv()

from sharek_agents.agents.skill_profiling.router import (
    generate_from_selected_evidence,
    profile_contributor,
)
from sharek_agents.agents.skill_profiling.schemas import (
    Contributor,
    SkillProfileGenerationRequest,
    SkillProfileGenerationResponse,
)
from sharek_agents.agents.advisory_fit.endpoint import analyze_advisory_fit
from sharek_agents.agents.advisory_fit.schemas import AdvisoryFitInput, AdvisoryFitResult
from sharek_agents.agents.skill_gap_guidance.endpoint import (
    analyze_skill_gap_guidance,
    stream_skill_gap_guidance,
)
from sharek_agents.agents.skill_gap_guidance.schemas import (
    SkillGapGuidanceInput,
    SkillGapGuidanceResult,
)
from sharek_agents.agents.contributor_matching.endpoint import (
    analyze_contributor_matching,
)
from sharek_agents.agents.contributor_matching.schemas import (
    ContributorMatchingInput,
    ContributorMatchingResult,
)

app = FastAPI(title="Share-k AI Agents", version="0.1.0")
internal_auth = HTTPBearer(auto_error=False)


@app.exception_handler(RequestValidationError)
async def safe_validation_error(_request, _error: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": "Request validation failed"},
    )


def require_internal_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(internal_auth),
) -> None:
    expected_token = os.environ.get("AI_SERVICE_AUTH_TOKEN", "")
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service authentication is not configured",
        )
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not secrets.compare_digest(credentials.credentials, expected_token)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal service credentials",
        )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/skill-profiles/{username}", response_model=Contributor)
async def profile_by_username(
    username: str,
    _authenticated: None = Depends(require_internal_auth),
) -> Contributor:
    return await profile_contributor(username)


@app.post(
    "/skill-profiles/generate",
    response_model=SkillProfileGenerationResponse,
)
async def generate_skill_profile(
    request: SkillProfileGenerationRequest,
    _authenticated: None = Depends(require_internal_auth),
) -> SkillProfileGenerationResponse:
    return await generate_from_selected_evidence(request)


@app.post(
    "/advisory-fit/assess",
    response_model=AdvisoryFitResult,
    responses={
        401: {"description": "Missing or invalid service bearer token"},
        502: {"description": "Provider output was invalid"},
        503: {"description": "Service authentication is not configured"},
        504: {"description": "Provider timed out"},
    },
)
async def advisory_fit_assessment(
    request: AdvisoryFitInput,
    _authenticated: None = Depends(require_internal_auth),
) -> AdvisoryFitResult:
    return await analyze_advisory_fit(request)


@app.post(
    "/skill-gap-guidance/generate",
    response_model=SkillGapGuidanceResult,
    responses={
        401: {"description": "Missing or invalid service bearer token"},
        502: {"description": "Provider output was invalid"},
        503: {"description": "Service authentication is not configured"},
        504: {"description": "Provider timed out"},
    },
)
async def skill_gap_guidance_endpoint(
    request: SkillGapGuidanceInput,
    _authenticated: None = Depends(require_internal_auth),
) -> SkillGapGuidanceResult:
    return await analyze_skill_gap_guidance(request)


@app.post(
    "/skill-gap-guidance/stream",
    responses={
        401: {"description": "Missing or invalid service bearer token"},
        502: {"description": "Provider output was invalid"},
        503: {"description": "Service authentication is not configured"},
        504: {"description": "Provider timed out"},
    },
)
async def skill_gap_guidance_stream_endpoint(
    request: SkillGapGuidanceInput,
    _authenticated: None = Depends(require_internal_auth),
):
    return await stream_skill_gap_guidance(request)


@app.post(
    "/contributor-matching/generate",
    response_model=ContributorMatchingResult,
    responses={
        401: {"description": "Missing or invalid service bearer token"},
        502: {"description": "Provider output was invalid"},
        503: {"description": "Service authentication is not configured"},
        504: {"description": "Provider timed out"},
    },
)
async def contributor_matching_endpoint(
    request: ContributorMatchingInput,
    _authenticated: None = Depends(require_internal_auth),
) -> ContributorMatchingResult:
    return await analyze_contributor_matching(request)
