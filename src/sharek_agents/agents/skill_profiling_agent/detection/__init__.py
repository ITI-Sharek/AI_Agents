"""Detection internals for the Detection Agent Tool (Phase 12).

Independent reimplementation of dependency-based technology detection:
registry (static knowledge + matching), manifest parsers, minimal GitHub
API access, and the deterministic detector pipeline. Nothing here imports
from the legacy ``skill_profiling.detection`` package.
"""

from sharek_agents.agents.skill_profiling_agent.detection.detector import (
    Detection,
    DetectionReport,
    FrameworksDetector,
)
from sharek_agents.agents.skill_profiling_agent.detection.github import (
    GitHubClient,
    RepositoryAccessError,
    RepositoryContext,
)
from sharek_agents.agents.skill_profiling_agent.detection.registry import (
    TECHNOLOGY_CATEGORIES,
    TechnologyEntry,
    TechnologyRegistry,
)

__all__ = [
    "TECHNOLOGY_CATEGORIES",
    "Detection",
    "DetectionReport",
    "FrameworksDetector",
    "GitHubClient",
    "RepositoryAccessError",
    "RepositoryContext",
    "TechnologyEntry",
    "TechnologyRegistry",
]
