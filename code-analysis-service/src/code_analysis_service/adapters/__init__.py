from __future__ import annotations

from ..models import StaticAnalysisEvidence
from .python_adapter import analyze_python
from .js_adapter import analyze_js
from .java_adapter import analyze_java
from .go_adapter import analyze_go
from .rust_adapter import analyze_rust
from .ruby_adapter import analyze_ruby
from .php_adapter import analyze_php
from .csharp_adapter import analyze_csharp
from .unsupported import analyze_unsupported


_ADAPTER_MAP: dict[str, callable] = {
    "python": analyze_python,
    "javascript": analyze_js,
    "typescript": analyze_js,
    "js": analyze_js,
    "ts": analyze_js,
    "java": analyze_java,
    "go": analyze_go,
    "rust": analyze_rust,
    "ruby": analyze_ruby,
    "php": analyze_php,
    "csharp": analyze_csharp,
    "c#": analyze_csharp,
}


def get_adapter(language: str) -> callable:
    adapter = _ADAPTER_MAP.get(language.lower())
    if adapter is not None:
        return adapter

    def _unsupported(
        repo_path: str, file_paths: list[str], timeout: int = 60
    ) -> StaticAnalysisEvidence:
        return StaticAnalysisEvidence(
            status="language_not_supported", language=language
        )

    return _unsupported


__all__ = [
    "get_adapter",
    "_ADAPTER_MAP",
    "analyze_python",
    "analyze_js",
    "analyze_java",
    "analyze_go",
    "analyze_rust",
    "analyze_ruby",
    "analyze_php",
    "analyze_csharp",
]
