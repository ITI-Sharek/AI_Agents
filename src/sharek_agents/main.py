import os
import secrets

from fastapi import Depends, FastAPI, HTTPException, status
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

app = FastAPI(title="Share-k AI Agents", version="0.1.0")
internal_auth = HTTPBearer(auto_error=False)


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
