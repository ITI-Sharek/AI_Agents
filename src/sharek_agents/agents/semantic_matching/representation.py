"""Semantic Matching canonical embedding-input representation (Phase 2).

The embedding model is NOT implemented in Phase 2 — no model calls, no API
calls, and no final embedding model is chosen here. This module only defines
the deterministic text representation that a future embedding service will
consume:

    Project/Contributor data
            |
            v
    Canonical text representation
            |
            v
    [Future Embedding Model]
            |
            v
    Vector
            |
            v
    pgvector

Determinism: the same source data always produces the same representation.
Skills are sorted by name (case-insensitive) with their level explicitly
attached (e.g. ``FastAPI: advanced``), so the relationship between a skill
and its level is never lost. Evidence is included when available, sorted by
title (case-insensitive) with stable field order. Arbitrary JSON
serialization is NOT used as the semantic representation.
"""

from __future__ import annotations

from sharek_agents.agents.semantic_matching.schemas import (
    ContributorSourceData,
    Evidence,
    ProjectSourceData,
    SkillItem,
)


EMBEDDING_REPRESENTATION_VERSION = "1"
"""Version of this canonical representation.

When embedding generation is implemented (later phase), the index record's
``embedding_schema_version`` metadata should be set to this constant, so
vectors built from a different representation version can be detected and
regenerated.
"""


def _sorted_skills(skills: list[SkillItem]) -> list[SkillItem]:
    return sorted(skills, key=lambda item: item.skill.casefold())


def _sorted_evidence(evidence: list[Evidence]) -> list[Evidence]:
    return sorted(evidence, key=lambda item: item.title.casefold())


def _sorted_technologies(technologies: list[str]) -> list[str]:
    return sorted(technologies, key=str.casefold)


def build_embedding_input(
    data: ProjectSourceData | ContributorSourceData,
) -> str:
    """Build the deterministic canonical representation for one entity.

    Args:
        data: Authoritative Project or Contributor source data.

    Returns:
        A stable multi-line text representation of skills + levels and
        relevant evidence. The same ``data`` always yields the same string.
    """
    lines = ["SKILLS"]
    skills = _sorted_skills(data.skills)
    if skills:
        lines.extend(f"{item.skill}: {item.level}" for item in skills)
    else:
        lines.append("(none)")

    evidence = _sorted_evidence(data.evidence)
    if evidence:
        lines.append("")
        lines.append("EVIDENCE")
        for item in evidence:
            lines.append(f"- {item.title}")
            if item.summary:
                lines.append(f"  Summary: {item.summary}")
            if item.technologies:
                technologies = ", ".join(_sorted_technologies(item.technologies))
                lines.append(f"  Technologies: {technologies}")
            if item.description:
                lines.append(f"  Description: {item.description}")

    return "\n".join(lines) + "\n"