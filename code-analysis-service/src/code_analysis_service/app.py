from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .adapters import get_adapter
from .graphify_runner import run_graphify
from .models import AnalysisResult, GraphRelationsEvidence, StaticAnalysisEvidence
from .orchestrator import analyze_repo as _analyze_repo_orchestrated
from .security import require_service_token


class AnalyzeRepoRequest(BaseModel):
    repo_url: str
    language: str
    requested_tools: list[Literal["static_analysis", "graph_relations"]]
    pat: str | None = None


class AnalyzeStaticRequest(BaseModel):
    repo_path: str
    language: str
    file_paths: list[str]


class AnalyzeGraphRequest(BaseModel):
    repo_path: str


app = FastAPI(title="Code Analysis API")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post(
    "/analyze/repo",
    response_model=AnalysisResult,
    dependencies=[Depends(require_service_token)],
)
async def analyze_repo_endpoint(body: AnalyzeRepoRequest):
    return await _analyze_repo_orchestrated(
        repo_url=body.repo_url,
        pat=body.pat,
        language=body.language,
        requested_tools=body.requested_tools,
    )


@app.post(
    "/analyze/static",
    response_model=StaticAnalysisEvidence,
    dependencies=[Depends(require_service_token)],
)
async def analyze_static_endpoint(body: AnalyzeStaticRequest):
    adapter = get_adapter(body.language)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, adapter, body.repo_path, body.file_paths, 60
    )


@app.post(
    "/analyze/graph",
    response_model=GraphRelationsEvidence,
    dependencies=[Depends(require_service_token)],
)
async def analyze_graph_endpoint(body: AnalyzeGraphRequest):
    return await run_graphify(
        cloned_repo_path=body.repo_path, timeout_seconds=60
    )
