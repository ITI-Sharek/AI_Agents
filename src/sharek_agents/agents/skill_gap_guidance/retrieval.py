from __future__ import annotations

import re
from dataclasses import dataclass

from sharek_agents.agents.skill_gap_guidance.schemas import SkillGapGuidanceInput


@dataclass(frozen=True)
class CuratedLearningResource:
    resource_id: str
    title: str
    resource_type: str
    url: str
    summary: str
    tags: frozenset[str]


CURATED_LEARNING_RESOURCES: tuple[CuratedLearningResource, ...] = (
    CuratedLearningResource(
        resource_id="airflow-official-docs",
        title="Apache Airflow documentation",
        resource_type="documentation",
        url="https://airflow.apache.org/docs/",
        summary="Official guides for DAG authoring, scheduling, and operations.",
        tags=frozenset({"airflow", "apache", "dag", "orchestration", "pipeline"}),
    ),
    CuratedLearningResource(
        resource_id="docker-get-started",
        title="Docker Get Started",
        resource_type="tutorial",
        url="https://docs.docker.com/get-started/",
        summary="Official container, image, and Compose fundamentals.",
        tags=frozenset({"docker", "container", "containers", "compose"}),
    ),
    CuratedLearningResource(
        resource_id="python-official-tutorial",
        title="Python tutorial",
        resource_type="documentation",
        url="https://docs.python.org/3/tutorial/",
        summary="The official Python language tutorial and core concepts.",
        tags=frozenset({"python"}),
    ),
    CuratedLearningResource(
        resource_id="pandas-getting-started",
        title="pandas getting started tutorials",
        resource_type="tutorial",
        url="https://pandas.pydata.org/docs/getting_started/intro_tutorials/",
        summary="Official tutorials for tabular data work with pandas.",
        tags=frozenset({"pandas", "dataframe", "data", "python"}),
    ),
    CuratedLearningResource(
        resource_id="postgresql-tutorial",
        title="PostgreSQL tutorial",
        resource_type="documentation",
        url="https://www.postgresql.org/docs/current/tutorial.html",
        summary="Official relational database and SQL tutorial.",
        tags=frozenset({"postgresql", "postgres", "sql", "database"}),
    ),
    CuratedLearningResource(
        resource_id="react-learn",
        title="React Learn",
        resource_type="documentation",
        url="https://react.dev/learn",
        summary="Official React concepts and interactive learning path.",
        tags=frozenset({"react", "javascript", "frontend", "ui"}),
    ),
    CuratedLearningResource(
        resource_id="typescript-handbook",
        title="TypeScript Handbook",
        resource_type="documentation",
        url="https://www.typescriptlang.org/docs/handbook/intro.html",
        summary="Official TypeScript language and type-system handbook.",
        tags=frozenset({"typescript", "javascript", "types"}),
    ),
    CuratedLearningResource(
        resource_id="nodejs-learn",
        title="Node.js Learn",
        resource_type="documentation",
        url="https://nodejs.org/en/learn/getting-started/introduction-to-nodejs",
        summary="Official introduction to Node.js runtime fundamentals.",
        tags=frozenset({"node", "nodejs", "javascript", "backend"}),
    ),
    CuratedLearningResource(
        resource_id="nestjs-documentation",
        title="NestJS documentation",
        resource_type="documentation",
        url="https://docs.nestjs.com/",
        summary="Official NestJS architecture and application-building guides.",
        tags=frozenset({"nestjs", "typescript", "node", "backend"}),
    ),
    CuratedLearningResource(
        resource_id="fastapi-tutorial",
        title="FastAPI tutorial",
        resource_type="documentation",
        url="https://fastapi.tiangolo.com/tutorial/",
        summary="Official FastAPI tutorial for typed Python APIs.",
        tags=frozenset({"fastapi", "python", "api", "backend"}),
    ),
    CuratedLearningResource(
        resource_id="git-book",
        title="Pro Git",
        resource_type="book",
        url="https://git-scm.com/book/en/v2",
        summary="The freely available Git reference book.",
        tags=frozenset({"git", "github", "version", "control"}),
    ),
)


def retrieve_curated_resources(
    input_data: SkillGapGuidanceInput,
    *,
    limit: int = 8,
) -> tuple[CuratedLearningResource, ...]:
    """Return only bounded catalog entries relevant to the fixed snapshot."""
    query = " ".join(
        [
            *(requirement.text for requirement in input_data.requirements),
            *(skill.name for skill in input_data.approved_skills),
        ]
    ).lower()
    query_tokens = set(re.findall(r"[a-z0-9]+", query))
    ranked: list[tuple[int, CuratedLearningResource]] = []
    for resource in CURATED_LEARNING_RESOURCES:
        score = len(query_tokens.intersection(resource.tags))
        if score:
            ranked.append((score, resource))
    ranked.sort(key=lambda item: (-item[0], item[1].resource_id))
    return tuple(resource for _, resource in ranked[:limit])
