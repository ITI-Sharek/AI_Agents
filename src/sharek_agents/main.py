from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from sharek_agents.agents.advisory_fit.endpoint import (
    analyze_advisory_fit,
)
from sharek_agents.agents.advisory_fit.schemas import (
    AdvisoryFitInput,
    AdvisoryFitResult,
)
from sharek_agents.agents.document_understanding.endpoint import (
    analyze_document as doc_understanding_analyze,
)
from sharek_agents.agents.document_understanding.schemas import (
    DocumentUnderstandingInput,
    DocumentUnderstandingResult,
)
from sharek_agents.agents.material_analysis_dev import (
    MaterialAnalysisInput,
    MaterialAnalysisResult,
    analyze_materials_dev_endpoint,
)
from sharek_agents.agents.gap_guidance.endpoint import (
    analyze_gap_guidance,
)
from sharek_agents.agents.gap_guidance.schemas import (
    GapGuidanceInput,
    GapGuidanceResult,
)
from sharek_agents.agents.skill_gap_guidance.endpoint import (
    analyze_skill_gap_guidance,
    stream_skill_gap_guidance,
)
from sharek_agents.agents.skill_gap_guidance.schemas import (
    SkillGapGuidanceInput,
    SkillGapGuidanceResult,
)
from sharek_agents.agents.semantic_matching.endpoint import (
    match_projects as semantic_matching_match,
)
from sharek_agents.agents.semantic_matching.schemas import (
    SemanticMatchRequest,
    SemanticMatchResponse,
)
from sharek_agents.agents.skill_profiling.contract_schemas import (
    SkillProfileInput,
    SkillProfileResult,
)
from sharek_agents.agents.skill_profiling.contract_service import (
    SkillProfileProviderError,
    SkillProfileProviderTimeout,
    generate_skill_profile,
)
from sharek_agents.agents.skill_profiling.router import profile_repos
from sharek_agents.agents.skill_profiling.schemas import AgentResponse
from sharek_agents.agents.skill_profiling_agent.endpoint import (
    generate_skill_profile_agent_endpoint,
)
from sharek_agents.agents.skill_profiling_agent.schemas import (
    SkillProfileAgentResponse,
)
from sharek_agents.common.logging import get_logger
from sharek_agents.security import require_service_token

logger = get_logger(__name__)

app = FastAPI(title="SHARE-K AI Agents")


class ProfileRequest(BaseModel):
    repo_urls: list[str] = Field(description="List of GitHub repository URLs")
    github_username: str = Field(description="GitHub username for commit filtering")


@app.post("/profile/repos", response_model=AgentResponse)
async def profile_repos_endpoint(body: ProfileRequest):
    return await profile_repos(body.repo_urls, github_username=body.github_username)


@app.post(
    "/skill-profiles/generate",
    response_model=SkillProfileResult,
    dependencies=[Depends(require_service_token)],
    responses={
        401: {"description": "Missing or invalid service bearer token"},
        502: {"description": "Provider returned invalid output or failed"},
        503: {"description": "Service authentication is not configured"},
        504: {"description": "Provider timed out"},
    },
)
async def generate_skill_profile_endpoint(body: SkillProfileInput):
    try:
        return await generate_skill_profile(body)
    except SkillProfileProviderTimeout as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Skill-profile provider timed out",
        ) from exc
    except SkillProfileProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Skill-profile provider returned an invalid response",
        ) from exc


@app.post(
    "/skill-profiles/agent/generate",
    response_model=SkillProfileAgentResponse,
    dependencies=[Depends(require_service_token)],
    responses={
        401: {"description": "Missing or invalid service bearer token"},
        502: {"description": "Skill Profiling Agent returned an invalid response"},
        503: {"description": "Service authentication is not configured"},
        504: {"description": "Skill Profiling Agent timed out"},
    },
)
async def generate_skill_profile_agent_endpoint_route(body: SkillProfileInput):
    return await generate_skill_profile_agent_endpoint(body)


@app.post(
    "/material-analysis/analyze",
    response_model=MaterialAnalysisResult,
    dependencies=[Depends(require_service_token)],
    responses={
        401: {"description": "Missing or invalid service bearer token"},
        404: {"description": "Development material analysis is disabled"},
        422: {"description": "Selected material content could not be parsed"},
        502: {"description": "Development material analysis provider error"},
        503: {"description": "Service authentication is not configured"},
    },
)
async def material_analysis_endpoint(body: MaterialAnalysisInput):
    return await analyze_materials_dev_endpoint(body)


@app.post(
    "/document-understanding/analyze",
    response_model=DocumentUnderstandingResult,
    dependencies=[Depends(require_service_token)],
    responses={
        401: {"description": "Missing or invalid service bearer token"},
        502: {"description": "Document processing or provider error"},
        503: {"description": "Service authentication is not configured"},
        504: {"description": "Provider timed out"},
    },
)
async def document_understanding_endpoint(body: DocumentUnderstandingInput):
    return await doc_understanding_analyze(body)


@app.post(
    "/advisory-fit/analyze",
    response_model=AdvisoryFitResult,
    dependencies=[Depends(require_service_token)],
    responses={
        401: {"description": "Missing or invalid service bearer token"},
        502: {"description": "Advisory Fit provider returned an invalid response"},
        503: {"description": "Service authentication is not configured"},
        504: {"description": "Advisory Fit provider timed out"},
    },
)
async def advisory_fit_endpoint(body: AdvisoryFitInput):
    return await analyze_advisory_fit(body)


@app.post(
    "/gap-guidance/generate",
    response_model=SkillGapGuidanceResult,
    dependencies=[Depends(require_service_token)],
    responses={
        401: {"description": "Missing or invalid service bearer token"},
        502: {"description": "Skill-gap guidance provider returned an invalid response"},
        503: {"description": "Service authentication is not configured"},
        504: {"description": "Skill-gap guidance provider timed out"},
    },
)
async def gap_guidance_endpoint(body: SkillGapGuidanceInput):
    return await analyze_skill_gap_guidance(body)


@app.post(
    "/skill-gap-guidance/generate",
    response_model=SkillGapGuidanceResult,
    dependencies=[Depends(require_service_token)],
    responses={
        401: {"description": "Missing or invalid service bearer token"},
        502: {"description": "Skill-gap guidance provider returned an invalid response"},
        503: {"description": "Service authentication is not configured"},
        504: {"description": "Skill-gap guidance provider timed out"},
    },
)
async def skill_gap_guidance_endpoint(body: SkillGapGuidanceInput):
    return await analyze_skill_gap_guidance(body)


@app.post(
    "/skill-gap-guidance/stream",
    dependencies=[Depends(require_service_token)],
    responses={
        401: {"description": "Missing or invalid service bearer token"},
        502: {"description": "Skill-gap guidance provider returned an invalid response"},
        503: {"description": "Service authentication is not configured"},
        504: {"description": "Skill-gap guidance provider timed out"},
    },
)
async def skill_gap_guidance_stream_endpoint(body: SkillGapGuidanceInput):
    return await stream_skill_gap_guidance(body)


@app.post(
    "/semantic-matching/match",
    response_model=SemanticMatchResponse,
    dependencies=[Depends(require_service_token)],
    responses={
        401: {"description": "Missing or invalid service bearer token"},
        404: {"description": "Requested Contributor does not exist"},
        501: {"description": "Project -> Contributors direction is not implemented"},
        502: {"description": "Matching index, source data, or indexing failure"},
        503: {"description": "Service authentication or matching storage is not configured"},
    },
)
async def semantic_matching_match_endpoint(body: SemanticMatchRequest):
    return await semantic_matching_match(body)


@app.get("/health")
async def health():
    return {"status": "ok"}
