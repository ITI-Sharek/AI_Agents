from sharek_agents.agents.skill_profiling.graph import run as run_graph
from sharek_agents.agents.skill_profiling.schemas import (
    AgentResponse,
    Contributor,
    RawSkill,
    Skill,
    SkillProfilingResult,
    Source,
)
from sharek_agents.agents.skill_profiling.tools import gather_all_evidence


MIN_REPOS_FOR_PROFILE = 2


def _build_sources(evidence: dict, skill_type: str) -> list[Source]:
    sources: list[Source] = []
    for repo in evidence.get("repos", []):
        if skill_type == "github_stats":
            sources.append(Source(
                type="github_stats",
                detail=(
                    f"repo: {repo['name']}, "
                    f"files: {repo['files_evaluated']}, "
                    f"commit_count: {repo['commit_count']}"
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
    if raw_skill.evidence_type == "github_stats":
        if not any(
            not r.get("static_analysis", {}).get("skipped")
            for r in evidence.get("repos", [])
        ):
            return round(min(0.35 + 0.05 * repo_count, 0.6), 2)
        return round(min(0.5 + 0.05 * repo_count, 0.6), 2)

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
    if repo_count < MIN_REPOS_FOR_PROFILE:
        avg *= 0.5 + 0.5 * (repo_count / MIN_REPOS_FOR_PROFILE)
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


def _all_repos_no_current_files(evidence: dict) -> bool:
    repos = evidence.get("repos", [])
    if not repos:
        return False
    return all(
        r.get("static_analysis", {}).get("skipped")
        and r["static_analysis"].get("reason") == "no_current_files"
        for r in repos
    )


async def profile_contributor(username: str) -> Contributor:
    evidence = await gather_all_evidence(username)

    repos = evidence.get("repos", [])
    if not repos:
        return Contributor(
            username=username,
            status="needs_review",
            confidence=0.0,
            skills=[],
        )

    if _all_repos_no_current_files(evidence):
        return Contributor(
            username=username,
            status="needs_review",
            confidence=0.0,
            skills=[],
        )

    response = await run_graph(username, evidence=evidence)
    if response.status == "failed":
        return Contributor(
            username=username,
            status="needs_review",
            confidence=0.0,
            skills=[],
        )

    profiling = response.data
    skills = _build_skills(profiling, evidence)
    confidence = _compute_overall_confidence(skills, evidence)

    return Contributor(
        username=username,
        status="success",
        confidence=confidence,
        skills=skills,
    )
