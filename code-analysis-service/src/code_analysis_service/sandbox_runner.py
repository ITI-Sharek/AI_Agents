from __future__ import annotations

"""In-container analysis runner.

Executed INSIDE the ``code-analysis-runner`` container image:

    python -m code_analysis_service.sandbox_runner static <language> /workspace [--timeout N]
    python -m code_analysis_service.sandbox_runner graph /workspace [--timeout N]

The orchestrator mounts the sandbox workspace at ``/workspace`` and this
module performs the full static-analysis / Graphify execution there. The
resulting evidence is serialized as JSON on stdout and reconstructed by
the orchestrator on the host; the host never executes analysis tools
itself.

A non-zero exit code is a fail-closed signal — the orchestrator reports
the tool as ``tool_unavailable`` and never falls back to host execution.
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .adapters import get_adapter
from .graphify_runner import run_graphify
from .models import EXTENSIONS, GraphRelationsEvidence, StaticAnalysisEvidence


def run_static_analysis(
    language: str, repo_path: str, timeout: int = 60
) -> StaticAnalysisEvidence:
    """Mirror of the orchestrator's original static-analysis step:
    discover analyzable files and delegate to the language adapter."""
    adapter = get_adapter(language)
    exts = EXTENSIONS.get(language.lower(), [])
    all_files = [
        str(p)
        for p in sorted(Path(repo_path).rglob("*"))
        if p.is_file() and p.suffix in exts
    ]
    if not all_files:
        return StaticAnalysisEvidence(
            status="no_analyzable_content", language=language
        )
    return adapter(repo_path=repo_path, file_paths=all_files, timeout=timeout)


async def run_graph_relations(
    repo_path: str, timeout: int = 60
) -> GraphRelationsEvidence:
    """Run the external ``graphify`` binary inside the container and parse
    its output — exactly the behavior of ``graphify_runner.run_graphify``,
    which now executes inside the sandbox."""
    return await run_graphify(cloned_repo_path=repo_path, timeout_seconds=timeout)


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sandbox_runner")
    parser.add_argument("kind", choices=("static", "graph"))
    parser.add_argument("path")
    parser.add_argument("--language", default="")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)

    try:
        if args.kind == "static":
            evidence = run_static_analysis(
                args.language, args.path, timeout=args.timeout
            )
        else:
            evidence = await run_graph_relations(
                args.path, timeout=args.timeout
            )
        print(json.dumps(asdict(evidence), default=str))
        return 0
    except Exception as exc:
        # Fail closed: emit an error evidence and exit non-zero so the
        # orchestrator reports tool_unavailable (never a host fallback).
        print(
            json.dumps(
                asdict(
                    StaticAnalysisEvidence(
                        status="error",
                        language=args.language,
                        error_message=str(exc)[:500],
                    )
                ),
                default=str,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
