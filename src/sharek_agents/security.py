import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from sharek_agents.config import settings


service_bearer = HTTPBearer(
    auto_error=False,
    scheme_name="AIServiceBearer",
    description="Internal bearer token shared with the Share-k backend.",
)


async def require_service_token(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(service_bearer),
    ],
) -> None:
    configured_token = settings.ai_service_auth_token
    if len(configured_token) < 32:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service authentication is not configured",
        )
    if (
        credentials is None
        or credentials.scheme.casefold() != "bearer"
        or not secrets.compare_digest(credentials.credentials, configured_token)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
