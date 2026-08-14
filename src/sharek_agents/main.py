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
from sharek_agents.agents.requirement_inference.endpoint import (
    analyze_requirement_inference,
)
from sharek_agents.agents.requirement_inference.schemas import (
    RequirementInferenceInput,
    RequirementInferenceResult,
)
from sharek_agents.agents.matching_rank.endpoint import analyze_matching_rank
from sharek_agents.agents.matching_rank.schemas import (
    MatchingRankInput,
    MatchingRankResult,
)
from sharek_agents.agents.material_analysis.endpoint import analyze_materials
from sharek_agents.agents.material_analysis.schemas import (
    MaterialAnalysisInput,
    MaterialAnalysisResult,
)
from sharek_agents.agents.skill_gap_guidance.endpoint import (
    analyze_skill_gap_guidance,
    stream_skill_gap_guidance,
)
from sharek_agents.agents.skill_gap_guidance.schemas import (
    SkillGapGuidanceInput,
    SkillGapGuidanceResult,
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
    "/requirements/infer",
    response_model=RequirementInferenceResult,
    responses={
        401: {"description": "Missing or invalid service bearer token"},
        502: {"description": "Provider output was invalid"},
        503: {"description": "Service authentication is not configured"},
        504: {"description": "Provider timed out"},
    },
)
async def requirement_inference(
    request: RequirementInferenceInput,
    _authenticated: None = Depends(require_internal_auth),
) -> RequirementInferenceResult:
    """Name the skills and levels a Contribution Request demands.

    Findings about the work, never a verdict about a person: NestJS derives the
    eligibility decision from these rows (DEC-078, ADR 0015). The input schema
    has no contributor field and forbids extras, so this endpoint cannot be
    handed contributor data even by mistake.
    """
    return await analyze_requirement_inference(request)


@app.post(
    "/material-analysis/analyze",
    response_model=MaterialAnalysisResult,
    responses={
        401: {"description": "Missing or invalid service bearer token"},
        502: {"description": "Provider output was invalid"},
        503: {"description": "Service authentication is not configured"},
        504: {"description": "Provider timed out"},
    },
)
async def material_analysis(
    request: MaterialAnalysisInput,
    _authenticated: None = Depends(require_internal_auth),
) -> MaterialAnalysisResult:
    """Draft suggestions from owner-supplied Materials.

    Produces private, individually adoptable suggestions only; nothing here
    mutates or publishes a Project or a Contribution Request (DEC-039).
    """
    return await analyze_materials(request)


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
async def skill_gap_guidance(
    request: SkillGapGuidanceInput,
    _authenticated: None = Depends(require_internal_auth),
) -> SkillGapGuidanceResult:
    """Educational guidance for a contributor who asked for it.

    Source-attributed recommendations only: never an eligibility verdict, a
    rank, or a change to any Application (ADR 0014, DEC-076).
    """
    return await analyze_skill_gap_guidance(request)


@app.get(
    "/skill-gap-guidance/stream",
    responses={
        401: {"description": "Missing or invalid service bearer token"},
        503: {"description": "Service authentication is not configured"},
    },
)
async def skill_gap_guidance_stream(
    request: SkillGapGuidanceInput = Depends(),
    _authenticated: None = Depends(require_internal_auth),
):
    """Final-result SSE transport for the same guidance contract."""
    return await stream_skill_gap_guidance(request)


@app.post(
    "/matching/rank",
    response_model=MatchingRankResult,
    responses={
        401: {"description": "Missing or invalid service bearer token"},
        502: {"description": "Provider output was invalid"},
        503: {"description": "Service authentication is not configured"},
        504: {"description": "Provider timed out"},
    },
)
async def matching_rank(
    request: MatchingRankInput,
    _authenticated: None = Depends(require_internal_auth),
) -> MatchingRankResult:
    """Reorder a shortlist the backend already computed, and explain each match.

    The agent never discovers candidates and never scores them: it returns a
    permutation of what it was given plus one sentence each. Anything else is
    rejected here and again by the backend, which falls back to its own
    deterministic order rather than showing a contributor nothing (DEC-010).
    """
    return await analyze_matching_rank(request)
