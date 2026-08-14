import os

from sharek_agents.agents.skill_profiling.graph import run as run_graph
from sharek_agents.agents.skill_profiling.schemas import (
    AgentResponse,
    Contributor,
    FraudSignal,
    GeneratedSkillCandidate,
    RawSkill,
    RepositoryEvidenceCapsule,
    Skill,
    SkillProfileGenerationRequest,
    SkillProfileGenerationResponse,
    SkillProfilingResult,
    Source,
)
from sharek_agents.agents.skill_profiling.tools import gather_all_evidence


MIN_REPOS_FOR_PROFILE = 2
PROMPT_VERSION = "skill-profile-selected-repos-v1"
SCHEMA_VERSION = "skill-profile-result-v1"
SERVICE_VERSION = "ai-service-0.1.0"
DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "openai/gpt-oss-120b"


def _build_sources(evidence: dict, skill_type: str) -> list[Source]:
    sources: list[Source] = []
    for repo in evidence.get("repos", []):
        if skill_type == "github_stats":
            authorship = repo.get("authorship", {})
            sources.append(Source(
                type="github_stats",
                detail=(
                    f"evidence_id: {repo.get('evidence_id', 'unavailable')}, "
                    f"repo: {repo['name']}, "
                    f"contributor_commits: {authorship.get('totalCommits', 0)}, "
                    f"contributor_additions: {authorship.get('additions', 0)}"
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

    if evidence.get("source") == "backend_selected_repositories":
        repos = evidence.get("repos", [])
        authored_repos = [
            repo
            for repo in repos
            if repo.get("authorship", {}).get("contributionDetected")
        ]
        if not authored_repos:
            return 0.2

        repos_with_languages = sum(
            1 for repo in authored_repos if repo.get("languages")
        )
        total_commits = sum(
            int(repo.get("authorship", {}).get("totalCommits", 0))
            for repo in authored_repos
        )
        total_additions = sum(
            int(repo.get("authorship", {}).get("additions", 0))
            for repo in authored_repos
        )
        base = 0.45
        if repos_with_languages:
            base += 0.1
        if total_commits >= 5:
            base += 0.1
        if total_additions >= 100:
            base += 0.05
        if raw_skill.evidence_type in {"static_analysis", "graphify_relations"}:
            base += 0.05
        return round(min(base + 0.03 * max(len(authored_repos) - 1, 0), 0.85), 2)

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


async def generate_from_selected_evidence(
    request: SkillProfileGenerationRequest,
) -> SkillProfileGenerationResponse:
    evidence = _build_backend_selected_evidence(request)
    fraud_signals = _build_fraud_signals(request.selectedRepositories)
    evidence_quality = _compute_backend_evidence_quality(
        request.selectedRepositories,
        fraud_signals,
    )

    if not request.selectedRepositories:
        return _empty_generation_response(
            evidence_quality="weak",
            fraud_signals=fraud_signals,
        )

    response = await run_graph(request.githubLogin, evidence=evidence)
    if response.status == "failed" or response.data is None:
        raise RuntimeError("Skill profiling provider failed")

    candidates = _build_generated_candidates(
        response.data,
        request.selectedRepositories,
        evidence,
    )
    recommendation = (
        "pending_review"
        if candidates and evidence_quality != "weak"
        else "needs_more_evidence"
    )

    return SkillProfileGenerationResponse(
        skills=candidates,
        fraudSignals=fraud_signals,
        evidenceQuality=evidence_quality,
        recommendation=recommendation,
        provider=os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER),
        model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        promptVersion=PROMPT_VERSION,
        schemaVersion=SCHEMA_VERSION,
        serviceVersion=SERVICE_VERSION,
    )


def _build_backend_selected_evidence(
    request: SkillProfileGenerationRequest,
) -> dict:
    return {
        "username": request.githubLogin,
        "source": "backend_selected_repositories",
        "generation_id": request.generationId,
        "repos": [
            _repo_capsule_to_legacy_evidence(repo)
            for repo in request.selectedRepositories
        ],
    }


def _repo_capsule_to_legacy_evidence(repo: RepositoryEvidenceCapsule) -> dict:
    recent_commits = repo.commitSignals.get("recentCommits", [])
    authored_commit_count = max(
        repo.authorship.totalCommits,
        repo.authorship.recentCommitCount,
    )

    return {
        "name": repo.fullName,
        "html_url": repo.htmlUrl,
        "private": repo.private,
        "fork": repo.fork,
        "archived": repo.archived,
        "description": repo.description,
        "topics": repo.topics,
        "primary_language": repo.primaryLanguage,
        "languages": repo.languages,
        "technologies": repo.technologies,
        "readme_excerpt": repo.readmeExcerpt,
        "repo_statistics": repo.statistics,
        "commit_count": authored_commit_count,
        "files_evaluated": 0,
        "static_analysis": {
            "skipped": True,
            "reason": "backend_capsule_metadata_only",
        },
        "graph_relations": {
            "files_analyzed": [],
            "inherits": [],
            "calls": [],
        },
        "contribution_activity": repo.contributionActivity,
        "commit_signals": repo.commitSignals,
        "authorship": repo.authorship.model_dump(),
        "evidence_failures": repo.evidenceFailures,
        "evidence_id": repo.evidenceId,
    }


def _build_generated_candidates(
    profiling: SkillProfilingResult,
    repositories: list[RepositoryEvidenceCapsule],
    evidence: dict,
) -> list[GeneratedSkillCandidate]:
    valid_repositories = {repo.evidenceId: repo for repo in repositories}
    candidates: list[GeneratedSkillCandidate] = []

    for raw_skill in profiling.skills:
        evidence_ids = list(dict.fromkeys(
            evidence_id
            for evidence_id in raw_skill.evidence_ids
            if evidence_id in valid_repositories
        ))
        if not evidence_ids:
            continue
        scoped_repositories = [
            valid_repositories[evidence_id] for evidence_id in evidence_ids
        ]
        scoped_evidence = {
            **evidence,
            "repos": [
                _repo_capsule_to_legacy_evidence(repo)
                for repo in scoped_repositories
            ],
        }
        confidence = _compute_skill_confidence(raw_skill, scoped_evidence)
        sources = _build_sources(scoped_evidence, raw_skill.evidence_type)
        if not sources:
            continue
        candidates.append(
            GeneratedSkillCandidate(
                name=raw_skill.name,
                proficiency=_proficiency_from_confidence(confidence),
                confidence=confidence,
                evidenceIds=evidence_ids,
                evidenceSummary=_summarize_candidate_evidence(
                    raw_skill.supporting_evidence,
                    sources,
                ),
                limitations=[
                    "Generated from backend-selected repository capsules; admin review is required."
                ],
            )
        )

    return candidates


def _build_fraud_signals(
    repositories: list[RepositoryEvidenceCapsule],
) -> list[FraudSignal]:
    signals: list[FraudSignal] = []

    for repo in repositories:
        commit_count = max(
            repo.authorship.totalCommits,
            repo.authorship.recentCommitCount,
        )
        readme_only = bool(repo.readmeExcerpt) and not repo.languages

        if repo.authorship.githubLogin.lower() != repo.fullName.split("/", 1)[0].lower() and not repo.authorship.contributionDetected:
            signals.append(FraudSignal(
                code="connected_contributor_not_attributed",
                severity="high",
                message="No commits attributable to the connected GitHub login were found.",
                repositoryFullName=repo.fullName,
            ))
        elif not repo.authorship.contributionDetected:
            signals.append(FraudSignal(
                code="no_connected_contributor_activity",
                severity="high",
                message="Repository activity could not be attributed to the connected contributor.",
                repositoryFullName=repo.fullName,
            ))

        if repo.fork:
            signals.append(FraudSignal(
                code="repository_is_fork",
                severity="medium",
                message="Repository is a fork; authored contribution depth needs review.",
                repositoryFullName=repo.fullName,
            ))
        if commit_count == 0:
            signals.append(FraudSignal(
                code="no_recent_commit_signal",
                severity="medium",
                message="No recent commit signal was available for this repository.",
                repositoryFullName=repo.fullName,
            ))
        if readme_only:
            signals.append(FraudSignal(
                code="readme_only_signal",
                severity="low",
                message="README exists but language evidence is missing.",
                repositoryFullName=repo.fullName,
            ))
        if repo.evidenceFailures:
            signals.append(FraudSignal(
                code="partial_repository_evidence",
                severity="low",
                message="Some repository evidence sources were unavailable.",
                repositoryFullName=repo.fullName,
            ))

    return signals


def _compute_backend_evidence_quality(
    repositories: list[RepositoryEvidenceCapsule],
    fraud_signals: list[FraudSignal],
) -> str:
    if not repositories:
        return "weak"

    high_or_medium_signals = [
        signal for signal in fraud_signals if signal.severity in {"medium", "high"}
    ]
    repos_with_language = sum(1 for repo in repositories if repo.languages)
    repos_with_authorship = sum(
        1
        for repo in repositories
        if repo.authorship.contributionDetected
    )

    if not repos_with_language or not repos_with_authorship:
        return "weak"
    if high_or_medium_signals:
        return "medium"
    if (
        len(repositories) >= 2
        and repos_with_language >= 2
        and repos_with_authorship >= 2
    ):
        return "strong"
    return "medium"


def _proficiency_from_confidence(confidence: float) -> str:
    if confidence >= 0.85:
        return "advanced"
    if confidence >= 0.65:
        return "intermediate"
    return "beginner"


def _summarize_sources(sources: list[Source]) -> str:
    return " | ".join(source.detail for source in sources[:3])


def _summarize_candidate_evidence(
    supporting_evidence: list[str],
    sources: list[Source],
) -> str:
    concrete_claims = [claim.strip() for claim in supporting_evidence if claim.strip()]
    if concrete_claims:
        return " | ".join(concrete_claims[:3])
    return _summarize_sources(sources)


def _empty_generation_response(
    evidence_quality: str,
    fraud_signals: list[FraudSignal],
) -> SkillProfileGenerationResponse:
    return SkillProfileGenerationResponse(
        skills=[],
        fraudSignals=fraud_signals,
        evidenceQuality=evidence_quality,
        recommendation="needs_more_evidence",
        provider=os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER),
        model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        promptVersion=PROMPT_VERSION,
        schemaVersion=SCHEMA_VERSION,
        serviceVersion=SERVICE_VERSION,
    )
