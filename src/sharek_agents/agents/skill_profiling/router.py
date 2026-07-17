from sharek_agents.agents.skill_profiling.graph import run as run_graph
from sharek_agents.agents.skill_profiling.schemas import (
    AgentResponse,
    ErrorInfo,
    FrameworkSkill,
    SkillProfilingResult,
    Source,
)
from sharek_agents.agents.skill_profiling.tools import gather_all_evidence


def _validate_framework_skills(profiling: SkillProfilingResult) -> str | None:
    names = [fs.name.lower() for fs in profiling.framework_skills]
    if len(names) != len(set(names)):
        return "framework_skills contains duplicate framework names (case-insensitive)"
    return None


def _apply_confidence_cap(framework_skills: list[FrameworkSkill]) -> list[FrameworkSkill]:
    for fs in framework_skills:
        if fs.evidence_type == "github_stats" and fs.confidence > 0.6:
            fs.confidence = 0.6
    return framework_skills


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


def _build_framework_sources(framework_skills: list[FrameworkSkill]) -> list[Source]:
    return [
        Source(detail=fs.evidence)
        for fs in framework_skills
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
        clean_code=response.clean_code,
        software_design_architecture=response.software_design_architecture,
        code_quality_maintainability=response.code_quality_maintainability,
        implementation=response.implementation,
        testing_practices=response.testing_practices,
        framework_skills=response.framework_skills or [],
    )

    fs_validation_error = _validate_framework_skills(profiling)
    if fs_validation_error:
        return AgentResponse(
            status="failed",
            error=ErrorInfo(
                code="invalid_profiling_output",
                message=fs_validation_error,
                retryable=False,
            ),
        )

    _apply_confidence_cap(profiling.framework_skills)

    all_sources = _build_evidence_sources(evidence)

    for gs in [
        profiling.clean_code,
        profiling.software_design_architecture,
        profiling.code_quality_maintainability,
        profiling.implementation,
        profiling.testing_practices,
    ]:
        for ev in gs.evidence:
            all_sources.append(Source(detail=ev))

    all_sources.extend(_build_framework_sources(profiling.framework_skills))

    return AgentResponse(
        status="success",
        clean_code=profiling.clean_code,
        software_design_architecture=profiling.software_design_architecture,
        code_quality_maintainability=profiling.code_quality_maintainability,
        implementation=profiling.implementation,
        testing_practices=profiling.testing_practices,
        framework_skills=profiling.framework_skills,
        confidence=1.0,
        sources=all_sources,
        unresolved_repos=unresolved_repos,
    )
