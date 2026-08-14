"""Framework/ORM/template/testing/library detection for the Detection Agent Tool.

The detector:

1. resolves the repository branch (argument, request context, then API),
2. discovers dependency manifest files through the GitHub API,
3. fetches and parses each discovered file in memory,
4. matches extracted package names against the static registry,
5. returns a deterministic, structured detection report.

It performs no LLM reasoning and no clone; parser failures for one file
never prevent other files from being analyzed. Unknown packages are
ignored. An explicit empty result is returned when no dependency files
exist rather than an error.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Literal, cast

from sharek_agents.agents.skill_profiling_agent.detection.github import (
    GitHubClient,
    RepositoryAccessError,
    RepositoryContext,
)
from sharek_agents.agents.skill_profiling_agent.detection.parsers import parser_for_path
from sharek_agents.agents.skill_profiling_agent.detection.registry import (
    TechnologyRegistry,
)

logger = logging.getLogger(__name__)

Category = Literal["framework", "orm", "template_engine", "testing", "library"]


@dataclass(frozen=True)
class Detection:
    """One detected technology tied to the package and file that matched."""

    name: str
    category: Category
    matched_package: str
    source_file: str


@dataclass(frozen=True)
class DetectionReport:
    """Structured detection evidence returned to the ReAct agent."""

    repository: str
    detections: list[Detection] = field(default_factory=list)
    dependency_files_found: list[str] = field(default_factory=list)
    detection_count: int = 0
    status: Literal["success", "no_dependency_files"] = "success"
    diagnostics: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "repository": self.repository,
                "status": self.status,
                "dependency_files_found": list(self.dependency_files_found),
                "detection_count": self.detection_count,
                "detections": [
                    {
                        "name": detection.name,
                        "category": detection.category,
                        "matched_package": detection.matched_package,
                        "source_file": detection.source_file,
                    }
                    for detection in self.detections
                ],
                "diagnostics": list(self.diagnostics),
            },
            ensure_ascii=False,
        )


class FrameworksDetector:
    """Deterministic dependency-based technology detector.

    ``registry`` and ``client`` are injectable so tests can control both
    without network access.
    """

    def __init__(
        self,
        *,
        context: RepositoryContext,
        registry: TechnologyRegistry | None = None,
        client: GitHubClient | None = None,
    ) -> None:
        self._context = context
        self._registry = registry or TechnologyRegistry()
        self._client = client or GitHubClient(context.github_token)

    @property
    def registry(self) -> TechnologyRegistry:
        return self._registry

    async def detect(
        self,
        repository: str,
        *,
        branch: str | None = None,
    ) -> DetectionReport:
        """Detect technologies used in one repository."""
        full_name = repository.strip().strip("/")
        if "/" not in full_name:
            raise ValueError(
                "repository must be an 'owner/name' GitHub repository identifier"
            )

        resolved_branch = await self._resolve_branch(full_name, branch)
        candidates = _candidate_file_names()
        file_paths = await self._client.list_dependency_file_paths(
            full_name, resolved_branch, candidates
        )

        if not file_paths:
            return DetectionReport(
                repository=full_name,
                status="no_dependency_files",
                dependency_files_found=[],
            )

        detections: list[Detection] = []
        diagnostics: list[str] = []
        matched_names: set[str] = set()

        for path in file_paths:
            parser = parser_for_path(path)
            if parser is None:
                continue
            try:
                content = await self._client.get_file_content(full_name, path)
            except RepositoryAccessError as exc:
                diagnostics.append(f"Could not read {path}: {exc}")
                continue
            try:
                packages = parser(content)
            except Exception as exc:  # noqa: BLE001 - one bad file must not fail all
                diagnostics.append(
                    f"Could not parse {path}: {type(exc).__name__}"
                )
                continue
            for package in sorted(packages):
                for entry in self._registry.match(package):
                    if entry.name in matched_names:
                        continue
                    matched_names.add(entry.name)
                    detections.append(
                        Detection(
                            name=entry.name,
                            category=cast(Category, entry.category),
                            matched_package=package,
                            source_file=path,
                        )
                    )

        detections.sort(key=lambda d: (d.name, d.category))
        return DetectionReport(
            repository=full_name,
            detections=detections,
            dependency_files_found=file_paths,
            detection_count=len(detections),
            diagnostics=diagnostics,
        )

    async def _resolve_branch(
        self,
        full_name: str,
        branch: str | None,
    ) -> str:
        if branch and branch.strip():
            return branch.strip()
        known = self._context.default_branches or {}
        known_branch = known.get(full_name)
        if known_branch:
            return known_branch
        return await self._client.get_default_branch(full_name)


def _candidate_file_names() -> frozenset[str]:
    """Basenames of manifest files the detector knows how to parse."""
    names = {
        "requirements.txt",
        "pyproject.toml",
        "Pipfile",
        "package.json",
        "composer.json",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "go.mod",
        "Cargo.toml",
        "Gemfile",
        "pubspec.yaml",
        "Podfile",
        "Cartfile",
        "Directory.Packages.props",
        "packages.config",
    }
    return frozenset(names)
