from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from sharek_agents.agents.skill_profiling.contract_schemas import (
    SkillProfileInput,
    SkillProfileResult,
)
from sharek_agents.agents.skill_profiling.contract_service import (
    SkillProfileProviderError,
    SkillProfileProviderTimeout,
    generate_skill_profile,
)
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
from sharek_agents.agents.skill_profiling.router import profile_repos
from sharek_agents.agents.skill_profiling.schemas import AgentResponse
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


@app.get("/health")
async def health():
    return {"status": "ok"}
