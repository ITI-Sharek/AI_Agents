import asyncio
import json

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
from sharek_agents.common.llm import (
    OpenRouterLLM,
    StudentGatewayLLM,
    clear_cache,
    get_llm,
)
from sharek_agents.config import settings
from sharek_agents.main import app


GATEWAY_ANALYSIS = {
    "skills": [
        {
            "name": "TypeScript",
            "proficiency": "intermediate",
            "confidence": 0.8,
            "evidenceIds": ["github:sharek-dev/repo"],
            "evidenceSummary": "Authored TypeScript changes are present.",
            "limitations": [],
        }
    ],
    "fraudSignals": [],
}


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


def test_student_gateway_returns_validated_structured_output(monkeypatch) -> None:
    async def post(_client, url, **_kwargs):
        return httpx.Response(
            200,
            json={"output_text": json.dumps(GATEWAY_ANALYSIS)},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    client = StudentGatewayLLM(
        model="test-model",
        api_key="test-key",
        base_url="https://gateway.example",
    )

    result = asyncio.run(
        client.generate_structured(
            system_prompt="Assess repository evidence.",
            user_prompt="Evidence: TypeScript",
            response_model=ModelSkillProfileAnalysis,
        )
    )

    assert result == ModelSkillProfileAnalysis.model_validate(GATEWAY_ANALYSIS)


def test_get_llm_uses_the_configured_openrouter_provider(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ai_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "openrouter-test-key")
    monkeypatch.setattr(settings, "openrouter_model", "openrouter/test-model")
    clear_cache()

    client = get_llm()

    assert isinstance(client, OpenRouterLLM)
    assert client.max_retries == 0


def test_openrouter_returns_validated_structured_output(monkeypatch) -> None:
    async def post(_client, url, **_kwargs):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(GATEWAY_ANALYSIS)}}
                ]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    client = OpenRouterLLM(
        model="openrouter/test-model",
        api_key="openrouter-test-key",
    )

    result = asyncio.run(
        client.generate_structured(
            system_prompt="Assess repository evidence.",
            user_prompt="Evidence: TypeScript",
            response_model=ModelSkillProfileAnalysis,
        )
    )

    assert result == ModelSkillProfileAnalysis.model_validate(GATEWAY_ANALYSIS)


def test_skill_profile_endpoint_uses_snapshot_when_clone_analysis_is_unavailable(
    monkeypatch,
) -> None:
    original_post = httpx.AsyncClient.post

    async def post(client, url, **kwargs):
        absolute_url = client.base_url.join(url)
        if absolute_url.host == "gateway.test":
            return httpx.Response(
                200,
                json={"output_text": json.dumps(GATEWAY_ANALYSIS)},
                request=httpx.Request("POST", absolute_url),
            )
        return await original_post(client, url, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    token = "internal-test-token-that-is-long-enough"
    monkeypatch.setattr(settings, "ai_service_auth_token", token)
    monkeypatch.setattr(settings, "analysis_service_enabled", False)
    monkeypatch.setattr(settings, "ai_provider", "student-api-gateway")
    monkeypatch.setattr(settings, "getaway_base_url", "https://gateway.test")
    monkeypatch.setattr(settings, "getaway_iti_key", "gateway-test-key")
    monkeypatch.setattr(settings, "getaway_model", "gateway-test-model")

    payload = make_request(make_repository()).model_dump(mode="json", by_alias=True)
    repository = payload["selectedRepositories"][0]
    repository["fullName"] = "repository"
    repository["staticAnalysis"] = {
        "toolUsed": "code-analysis-engine",
        "status": "tool_unavailable",
    }

    response = post_json(
        "/skill-profiles/generate",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
    )

    assert response.status_code == 200, response.text
    assert response.json()["skills"][0]["name"] == "TypeScript"
    assert response.json()["provider"] == "student-api-gateway"
    assert response.json()["model"] == "gateway-test-model"
