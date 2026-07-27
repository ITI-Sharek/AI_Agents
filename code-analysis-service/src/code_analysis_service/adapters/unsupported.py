from __future__ import annotations

from ..models import StaticAnalysisEvidence


def analyze_unsupported(
    language: str,
    repo_path: str = "",
    file_paths: list[str] | None = None,
    timeout: int = 60,
) -> StaticAnalysisEvidence:
    return StaticAnalysisEvidence(
        status="language_not_supported", language=language
    )
