import asyncio

import httpx
import pytest

from sharek_agents.agents.skill_profiling import contract_service
from sharek_agents.agents.skill_profiling.contract_schemas import (
    FrameworkDetectionEvidence,
    GeneratedSkillCandidate,
    ModelSkillProfileAnalysis,
    RepositoryEvidenceCapsule,
    SkillProfileInput,
    SkillProfileResult,
)
from sharek_agents.agents.skill_profiling.contract_service import (
    SkillProfileProviderError,
    _deterministic_fraud_signals,
    _validate_model_citations,
    assess_evidence_quality,
    generate_skill_profile,
)
from sharek_agents.main import app


def post_json(path: str, *, headers: dict[str, str] | None = None, json=None):
    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(path, headers=headers, json=json)

    return asyncio.run(request())


def make_repository(
    *,
    contributed: bool = True,
    github_login: str = "sharek-dev",
) -> RepositoryEvidenceCapsule:
    return RepositoryEvidenceCapsule(
        evidenceId="github:sharek-dev/repo",
        fullName="sharek-dev/repo",
        htmlUrl="https://github.com/sharek-dev/repo",
        private=False,
        fork=False,
        archived=False,
        defaultBranch="main",
        owner="sharek-dev",
        description=None,
        topics=[],
        primaryLanguage="TypeScript",
        languages={"TypeScript": 1000},
        technologies=["TypeScript"],
        statistics={},
        readmeExcerpt=None,
        contributionActivity={},
        commitSignals={},
        authorship={
            "githubLogin": github_login,
            "repositoryOwned": True,
            "recentCommitCount": 2 if contributed else 0,
            "totalCommits": 6 if contributed else 0,
            "additions": 250 if contributed else 0,
            "deletions": 30 if contributed else 0,
            "contributionDetected": contributed,
            "matchedRecentCommitShas": ["abc"] if contributed else [],
        },
        evidenceFailures=[],
    )


def make_request(repository: RepositoryEvidenceCapsule) -> SkillProfileInput:
    return SkillProfileInput(
        contributorId="user-1",
        githubLogin="sharek-dev",
        generationId="generation-1",
        role="contributor",
        selectedRepositories=[repository],
        requestedAt="2026-07-14T00:00:00.000Z",
    )


def test_internal_endpoint_rejects_missing_credentials(monkeypatch) -> None:
    monkeypatch.setattr(
        "sharek_agents.security.settings.ai_service_auth_token",
        "internal-test-token-that-is-long-enough",
    )

    response = post_json("/skill-profiles/generate", json={})

    assert response.status_code == 401


def test_internal_endpoint_accepts_the_nestjs_contributor_contract(
    monkeypatch,
) -> None:
    token = "internal-test-token-that-is-long-enough"

    async def generated_profile(_request: SkillProfileInput) -> SkillProfileResult:
        return SkillProfileResult(
            skills=[],
            fraudSignals=[],
            evidenceQuality="weak",
            recommendation="needs_more_evidence",
            provider="test",
            model="test-model",
            promptVersion="test-prompt",
            schemaVersion="test-schema",
            serviceVersion="test-service",
        )

    monkeypatch.setattr(
        "sharek_agents.security.settings.ai_service_auth_token",
        token,
    )
    monkeypatch.setattr(
        "sharek_agents.main.generate_skill_profile",
        generated_profile,
    )

    response = post_json(
        "/skill-profiles/generate",
        headers={"Authorization": f"Bearer {token}"},
        json=make_request(make_repository()).model_dump(
            mode="json",
            by_alias=True,
        ),
    )

    assert response.status_code == 200


def test_legacy_request_defaults_to_contributor_and_allows_missing_login() -> None:
    payload = make_request(make_repository()).model_dump(
        mode="json",
        by_alias=True,
    )
    payload.pop("role")
    payload["githubLogin"] = ""
    payload["selectedRepositories"][0]["authorship"]["githubLogin"] = ""

    request = SkillProfileInput.model_validate(payload)

    assert request.role == "contributor"
    assert request.github_login == ""
    assert request.selected_repositories[0].authorship.github_login == ""


def test_unattributed_repository_produces_weak_evidence() -> None:
    repository = make_repository(contributed=False, github_login="")

    assert assess_evidence_quality([repository]) == "weak"


def test_model_citations_must_match_submitted_evidence_ids() -> None:
    request = make_request(make_repository())
    analysis = ModelSkillProfileAnalysis(
        skills=[
            GeneratedSkillCandidate(
                name="TypeScript",
                proficiency="intermediate",
                confidence=0.8,
                evidenceIds=["github:someone-else/repo"],
            )
        ]
    )

    with pytest.raises(SkillProfileProviderError, match="unknown evidence ID"):
        _validate_model_citations(analysis, request)


def test_login_mismatch_emits_a_high_severity_fraud_signal() -> None:
    request = make_request(make_repository(github_login="someone-else"))

    signals = _deterministic_fraud_signals(request)

    assert any(
        signal.code == "github_login_mismatch" and signal.severity == "high"
        for signal in signals
    )


def test_provider_failure_is_retryable_by_the_backend(monkeypatch) -> None:
    async def detected_frameworks(*_args, **_kwargs):
        return FrameworkDetectionEvidence(
            frameworksDetected={"NestJS": ["package.json"]},
            dependencyFilesIdentified=[],
            frameworksCount=1,
            status="success",
        )

    async def skipped_analysis(*_args, **_kwargs) -> None:
        return None

    async def failed_model(*_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        contract_service,
        "_run_step1_framework_detection",
        detected_frameworks,
    )
    monkeypatch.setattr(contract_service, "_run_step2_analysis", skipped_analysis)
    monkeypatch.setattr(contract_service, "_invoke_model", failed_model)

    with pytest.raises(SkillProfileProviderError, match="provider failed"):
        asyncio.run(generate_skill_profile(make_request(make_repository())))
