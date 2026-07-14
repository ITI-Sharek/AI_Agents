import asyncio
import sys

from fastapi import FastAPI
from pydantic import BaseModel, Field

from sharek_agents.agents.skill_profiling.router import profile_repos
from sharek_agents.agents.skill_profiling.schemas import AgentResponse
from sharek_agents.common.logging import get_logger


logger = get_logger(__name__)

app = FastAPI(title="SHARE-K AI Agents")


@app.on_event("startup")
async def check_external_tools():
    for tool, args in [
        ("graphify", [sys.executable, "-m", "graphify", "--version"]),
        ("git", ["git", "--version"]),
    ]:
        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(
                "TOOL MISSING: '%s' is not available — %s",
                tool,
                stderr.decode().strip() or stdout.decode().strip(),
            )
        else:
            logger.info("Tool found: %s", stdout.decode().strip())


class ProfileRequest(BaseModel):
    repo_urls: list[str] = Field(description="List of GitHub repository URLs")
    github_username: str = Field(description="GitHub username for commit filtering")


@app.post("/profile/repos", response_model=AgentResponse)
async def profile_repos_endpoint(body: ProfileRequest):
    return await profile_repos(body.repo_urls, github_username=body.github_username)


@app.get("/health")
async def health():
    return {"status": "ok"}
