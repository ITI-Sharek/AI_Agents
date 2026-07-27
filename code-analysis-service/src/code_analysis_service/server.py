from __future__ import annotations

import json
from dataclasses import asdict
from typing import Literal

from .adapters import get_adapter
from .graphify_runner import run_graphify
from .models import AnalysisResult
from .orchestrator import analyze_repo as _analyze_repo_orchestrated


def _serialize(obj: object) -> str:
    return json.dumps(asdict(obj), default=str)


def analyze_static(
    repo_path: str,
    language: str,
    file_paths: list[str],
    timeout: int = 60,
) -> str:
    adapter = get_adapter(language)
    evidence = adapter(
        repo_path=repo_path, file_paths=file_paths, timeout=timeout
    )
    return _serialize(evidence)


async def analyze_graph(
    repo_path: str,
    timeout: int = 60,
) -> str:
    evidence = await run_graphify(
        cloned_repo_path=repo_path, timeout_seconds=timeout
    )
    return _serialize(evidence)


async def analyze_repo(
    repo_url: str,
    language: str,
    requested_tools: list[Literal["static_analysis", "graph_relations"]],
    pat: str | None = None,
) -> str:
    result = await _analyze_repo_orchestrated(
        repo_url=repo_url,
        pat=pat,
        language=language,
        requested_tools=requested_tools,
    )
    return _serialize(result)
