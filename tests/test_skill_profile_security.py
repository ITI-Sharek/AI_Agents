import asyncio

import httpx
import pytest

from sharek_agents.agents.skill_profiling.router import (
    _build_backend_selected_evidence,
    _build_fraud_signals,
    _build_generated_candidates,
    _compute_backend_evidence_quality,
    _compute_skill_confidence,
    generate_from_selected_evidence,
)
from sharek_agents.agents.skill_profiling.schemas import (
    AgentResponse,
    RawSkill,
    RepositoryAuthorship,
    RepositoryEvidenceCapsule,
    SkillProfileGenerationRequest,
    SkillProfilingResult,
)
from sharek_agents.main import app


def make_repository(*, contributed: bool = True) -> RepositoryEvidenceCapsule:
    return RepositoryEvidenceCapsule(
        evidenceId="github:sharek-dev/repo",
        fullName="sharek-dev/repo",
        htmlUrl="https://github.com/sharek-dev/repo",
        private=False,
        fork=False,
        archived=False,
        defaultBranch="main",
        owner="sharek-dev",
        primaryLanguage="TypeScript",
        languages={"TypeScript": 1000},
        technologies=["TypeScript"],
        authorship=RepositoryAuthorship(
            githubLogin="sharek-dev",
            repositoryOwned=True,
            recentCommitCount=2 if contributed else 0,
            totalCommits=6 if contributed else 0,
            additions=250 if contributed else 0,
            deletions=30 if contributed else 0,
            contributionDetected=contributed,
            matchedRecentCommitShas=["abc"] if contributed else [],
        ),
    )


def make_request(repository: RepositoryEvidenceCapsule) -> SkillProfileGenerationRequest:
    return SkillProfileGenerationRequest(
        contributorId="user-1",
        githubLogin="sharek-dev",
        generationId="generation-1",
        selectedRepositories=[repository],
        requestedAt="2026-07-14T00:00:00.000Z",
    )


def test_internal_endpoint_rejects_missing_credentials(monkeypatch) -> None:
    monkeypatch.setenv("AI_SERVICE_AUTH_TOKEN", "internal-test-token")

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.post("/skill-profiles/generate", json={})

    response = asyncio.run(request())
    assert response.status_code == 401


def test_unattributed_repository_cannot_receive_passing_confidence() -> None:
    raw_skill = RawSkill(
        name="TypeScript",
        evidence_type="github_stats",
        description="TypeScript",
        supporting_evidence=["Repository uses TypeScript"],
        evidence_ids=["github:sharek-dev/repo"],
    )
    evidence = _build_backend_selected_evidence(make_request(make_repository(contributed=False)))

    assert _compute_skill_confidence(raw_skill, evidence) == 0.2


def test_candidate_keeps_only_exact_cited_evidence_ids() -> None:
    repository = make_repository()
    request = make_request(repository)
    profiling = SkillProfilingResult(
        skills=[
            RawSkill(
                name="TypeScript",
                evidence_type="github_stats",
                description="TypeScript",
                supporting_evidence=["Six contributor commits"],
                evidence_ids=[repository.evidenceId, "github:someone-else/repo"],
            )
        ],
        overall_level="Intermediate",
        summary="Profile",
    )
    evidence = _build_backend_selected_evidence(request)

    candidates = _build_generated_candidates(
        profiling,
        request.selectedRepositories,
        evidence,
    )

    assert len(candidates) == 1
    assert candidates[0].evidenceIds == [repository.evidenceId]
    assert candidates[0].confidence >= 0.7


def test_evidence_without_connected_contributor_activity_is_weak() -> None:
    repository = make_repository(contributed=False)
    assert _compute_backend_evidence_quality([repository], []) == "weak"


def test_recent_attributed_commit_does_not_emit_no_commit_signal() -> None:
    repository = make_repository()
    repository.authorship.totalCommits = 0
    repository.authorship.recentCommitCount = 1

    signals = _build_fraud_signals([repository])

    assert "no_recent_commit_signal" not in {signal.code for signal in signals}


def test_provider_failure_is_retryable_by_the_backend(monkeypatch) -> None:
    async def failed_graph(*_args, **_kwargs) -> AgentResponse:
        return AgentResponse(
            status="failed",
            error_code="llm_provider_error",
            retryable=True,
        )

    monkeypatch.setattr(
        "sharek_agents.agents.skill_profiling.router.run_graph",
        failed_graph,
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        asyncio.run(generate_from_selected_evidence(make_request(make_repository())))
