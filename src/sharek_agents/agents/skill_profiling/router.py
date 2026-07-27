from sharek_agents.agents.skill_profiling.graph import run as run_graph
from sharek_agents.agents.skill_profiling.schemas import (
    AgentResponse,
    ErrorInfo,
    Skill,
    SkillProfilingResult,
    Source,
)
from sharek_agents.agents.skill_profiling.tools import gather_all_evidence


def _validate_skills(profiling: SkillProfilingResult) -> str | None:
    names = [s.name.lower() for s in profiling.skills]
    if len(names) != len(set(names)):
        return "skills contains duplicate skill names (case-insensitive)"
    return None


def _apply_confidence_cap(skills: list[Skill]) -> list[Skill]:
    for s in skills:
        if s.evidence_type == "github_stats" and s.confidence > 0.6:
            s.confidence = 0.6
    return skills


def _build_evidence_sources(evidence: dict) -> list[Source]:
    sources: list[Source] = []
    for repo in evidence.get("repos", []):
        sources.append(Source(
            detail=f"repo: {repo['name']}, language: {repo['language']}, topics: {repo.get('topics', [])}"
        ))
        frameworks = repo.get("frameworks", {})
        if frameworks:
            sources.append(Source(
                detail=f"repo: {repo['name']}, frameworks: {frameworks}"
            ))
        sa = repo.get("static_analysis", {})
        if not sa.get("skipped"):
            sources.append(Source(
                detail=(
                    f"repo: {repo['name']}, "
                    f"MI: {sa.get('maintainability_index', 'N/A')}, "
                    f"pylint: {sa.get('pylint_score', 'N/A')}, "
                    f"avg CC: {sa.get('avg_complexity', 'N/A')}"
                )
            ))
        gr = repo.get("graph_relations", {})
        sources.append(Source(
            detail=(
                f"repo: {repo['name']}, "
                f"inherits: {len(gr.get('inherits', []))} edges, "
                f"calls: {len(gr.get('calls', []))} edges"
            )
        ))
    return sources


def _build_skill_sources(skills: list[Skill]) -> list[Source]:
    return [
        Source(detail=s.evidence)
        for s in skills
    ]


def _all_repos_no_source_files(evidence: dict) -> bool:
    repos = evidence.get("repos", [])
    if not repos:
        return False
    return all(
        r.get("static_analysis", {}).get("skipped")
        and r["static_analysis"].get("reason") == "no_source_files"
        for r in repos
    )


async def profile_repos(repo_urls: list[str], github_username: str) -> AgentResponse:
    evidence = await gather_all_evidence(repo_urls, github_username)

    repos = evidence.get("repos", [])
    unresolved_repos = evidence.get("unresolved_repos", [])

    if len(unresolved_repos) == len(repo_urls):
        return AgentResponse(
            status="failed",
            error=ErrorInfo(
                code="no_valid_repos",
                message="All provided repository URLs could not be resolved",
                retryable=False,
            ),
        )

    if not repos:
        return AgentResponse(
            status="failed",
            error=ErrorInfo(
                code="no_source_files",
                message="No repository evidence was collected for profiling",
                retryable=False,
            ),
        )

    if _all_repos_no_source_files(evidence):
        return AgentResponse(
            status="failed",
            error=ErrorInfo(
                code="no_source_files",
                message="All resolved repositories have no source files",
                retryable=False,
            ),
        )

    response = await run_graph("", evidence=evidence)
    if response.status == "failed":
        return AgentResponse(
            status="failed",
            error=response.error,
        )

    profiling = SkillProfilingResult(
        skills=response.skills or [],
    )

    validation_error = _validate_skills(profiling)
    if validation_error:
        return AgentResponse(
            status="failed",
            error=ErrorInfo(
                code="invalid_profiling_output",
                message=validation_error,
                retryable=False,
            ),
        )

    _apply_confidence_cap(profiling.skills)

    all_sources = _build_evidence_sources(evidence)
    all_sources.extend(_build_skill_sources(profiling.skills))

    return AgentResponse(
        status="success",
        skills=profiling.skills,
        confidence=1.0,
        sources=all_sources,
        unresolved_repos=unresolved_repos,
    )
