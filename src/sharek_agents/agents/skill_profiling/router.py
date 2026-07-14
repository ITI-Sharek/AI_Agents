from sharek_agents.agents.skill_profiling.graph import run as run_graph
from sharek_agents.agents.skill_profiling.schemas import (
    AgentResponse,
    ErrorInfo,
    GeneralSkill,
    RawSkill,
    RepoProfile,
    Skill,
    SkillProfilingResult,
    Source,
)
from sharek_agents.agents.skill_profiling.tools import gather_all_evidence


MIN_SOURCE_FILES_FOR_PROFILE = 2

GENERAL_SKILL_NAMES = [
    "Clean Code",
    "Software Design & Architecture",
    "Code Quality & Maintainability",
    "Implementation",
    "Testing Practices",
]


def _validate_profiling_result(profiling: SkillProfilingResult) -> str | None:
    skill_names_lower = [s.name.lower() for s in profiling.skills]
    if len(skill_names_lower) != len(set(skill_names_lower)):
        return "Duplicate skill names detected in skills list"

    if not profiling.general_skills:
        return "general_skills is missing from LLM output"
    if len(profiling.general_skills) != 5:
        return (
            f"general_skills has {len(profiling.general_skills)} entries, "
            f"expected exactly 5"
        )
    gs_names = [gs.name for gs in profiling.general_skills]
    if gs_names != GENERAL_SKILL_NAMES:
        return (
            f"general_skills names {gs_names} do not match required order: "
            f"{GENERAL_SKILL_NAMES}"
        )
    return None


def _build_sources(evidence: dict, skill_type: str) -> list[Source]:
    sources: list[Source] = []
    for repo in evidence.get("repos", []):
        if skill_type == "repo_metadata":
            sources.append(Source(
                type="repo_metadata",
                detail=(
                    f"repo: {repo['name']}, "
                    f"language: {repo['language']}, "
                    f"topics: {repo.get('topics', [])}, "
                    f"stars: {repo.get('stars', 0)}, "
                    f"forks: {repo.get('forks', 0)}"
                ),
            ))
        elif skill_type == "framework_detection":
            frameworks = repo.get("frameworks", {})
            if frameworks:
                sources.append(Source(
                    type="framework_detection",
                    detail=(
                        f"repo: {repo['name']}, "
                        f"frameworks: {frameworks}"
                    ),
                ))
        elif skill_type == "static_analysis":
            sa = repo.get("static_analysis", {})
            if not sa.get("skipped"):
                sources.append(Source(
                    type="static_analysis",
                    detail=(
                        f"repo: {repo['name']}, "
                        f"files: {sa.get('files_analyzed', [])}, "
                        f"maintainability_index: {sa.get('maintainability_index', 'N/A')}, "
                        f"pylint_score: {sa.get('pylint_score', 'N/A')}, "
                        f"avg_complexity: {sa.get('avg_complexity', 'N/A')}"
                    ),
                ))
        elif skill_type == "graphify_relations":
            gr = repo.get("graph_relations", {})
            sources.append(Source(
                type="graphify_relations",
                detail=(
                    f"repo: {repo['name']}, "
                    f"files: {gr.get('files_analyzed', [])}, "
                    f"inherits: {len(gr.get('inherits', []))} edges, "
                    f"calls: {len(gr.get('calls', []))} edges"
                ),
            ))
    return sources


def _compute_skill_confidence(
    raw_skill: RawSkill, evidence: dict
) -> float:
    repo_count = len(evidence.get("repos", []))
    if raw_skill.evidence_type == "repo_metadata":
        return round(min(0.5 + 0.05 * repo_count, 0.6), 2)

    if raw_skill.evidence_type == "framework_detection":
        return round(min(0.55 + 0.05 * repo_count, 0.7), 2)

    if raw_skill.evidence_type == "static_analysis":
        return round(min(0.65 + 0.05 * repo_count, 1.0), 2)

    return round(min(0.6 + 0.05 * repo_count, 1.0), 2)


def _compute_overall_confidence(
    skills: list[Skill], evidence: dict
) -> float:
    if not skills:
        return 0.0
    avg = sum(s.confidence for s in skills) / len(skills)
    repo_count = len(evidence.get("repos", []))
    if repo_count == 0:
        avg *= 0
    elif repo_count < MIN_SOURCE_FILES_FOR_PROFILE:
        avg *= 0.5 + 0.5 * (repo_count / MIN_SOURCE_FILES_FOR_PROFILE)
    return round(avg, 2)


def _build_skills(
    profiling: SkillProfilingResult, evidence: dict
) -> list[Skill]:
    return [
        Skill(
            name=s.name,
            confidence=_compute_skill_confidence(s, evidence),
            sources=_build_sources(evidence, s.evidence_type),
        )
        for s in profiling.skills
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

    profiling = response.data

    validation_error = _validate_profiling_result(profiling)
    if validation_error:
        return AgentResponse(
            status="failed",
            error=ErrorInfo(
                code="invalid_profiling_output",
                message=validation_error,
                retryable=False,
            ),
        )

    results: list[RepoProfile] = []
    for repo in repos:
        single_evidence = {
            "repo_count": 1,
            "repos": [repo],
        }
        skills = _build_skills(profiling, single_evidence)
        confidence = _compute_overall_confidence(skills, single_evidence)

        from sharek_agents.agents.skill_profiling.schemas import RepoInfo
        results.append(RepoProfile(
            repo=RepoInfo(
                name=repo["name"],
                owner=repo["owner"],
                description=repo["description"],
                language=repo["language"],
                topics=repo["topics"],
                stars=repo["stars"],
                forks=repo["forks"],
                clone_url=repo["clone_url"],
                default_branch=repo["default_branch"],
            ),
            status="success",
            confidence=confidence,
            skills=skills,
        ))

    all_sources: list[Source] = []
    seen_source_details: set[str] = set()
    for r in results:
        for s in r.skills:
            for src in s.sources:
                if src.detail not in seen_source_details:
                    seen_source_details.add(src.detail)
                    all_sources.append(src)

    for gs in profiling.general_skills:
        src = Source(type="general_skill", detail=gs.evidence)
        if src.detail not in seen_source_details:
            seen_source_details.add(src.detail)
            all_sources.append(src)

    repo_avg = (
        sum(r.confidence for r in results) / len(results)
        if results else 0.0
    )
    general_avg = (
        sum(g.confidence for g in profiling.general_skills) / len(profiling.general_skills)
        if profiling.general_skills else 0.0
    )

    if results and profiling.general_skills:
        overall_confidence = round((repo_avg + general_avg) / 2, 2)
    elif results:
        overall_confidence = round(repo_avg, 2)
    else:
        overall_confidence = round(general_avg, 2)

    return AgentResponse(
        status="success",
        data=results,
        general_skills=profiling.general_skills,
        confidence=overall_confidence,
        sources=all_sources,
        unresolved_repos=unresolved_repos,
    )
