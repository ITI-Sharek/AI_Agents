from datetime import datetime
import asyncio

import httpx
import pytest
from pydantic import ValidationError

from sharek_agents.agents.advisory_fit.schemas import (
    AdvisoryFitFinding,
    AdvisoryFitInput,
    AdvisoryFitMetadata,
    AdvisoryFitProviderOutput,
    AdvisoryFitResult,
)
from sharek_agents.agents.advisory_fit.service import (
    AdvisoryFitProviderError,
    AdvisoryFitProviderResponse,
    AdvisoryFitProviderSystemLimit,
    AdvisoryFitProviderTimeout,
    _invoke_provider,
    generate_advisory_fit,
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


def assessment_request_payload() -> dict:
    return {
        "assessmentRequestId": "00000000-0000-4000-8000-000000000001",
        "requirements": [
            {
                "id": "00000000-0000-4000-8000-000000000002",
                "kind": "required",
                "position": 0,
                "text": "NestJS",
            }
        ],
        "evidence": [
            {
                "skillProfileId": "00000000-0000-4000-8000-000000000003",
                "name": "NestJS",
                "evidenceSources": {"evidenceIds": ["github:evidence-1"]},
            }
        ],
        "allowedEvidenceIds": ["github:evidence-1"],
        "requestedAt": "2026-08-02T12:00:00.000Z",
        "contractVersion": "advisory-fit-v1",
    }


def test_accepts_the_backend_assessment_request_contract() -> None:
    request = AdvisoryFitInput.model_validate(assessment_request_payload())

    assert request.assessment_request_id.endswith("0001")
    assert request.requirements[0].kind == "required"
    assert request.allowed_evidence_ids == ["github:evidence-1"]
    assert request.requested_at == datetime.fromisoformat("2026-08-02T12:00:00+00:00")


def completed_provider_output() -> AdvisoryFitProviderOutput:
    return AdvisoryFitProviderOutput(
        findings=[
            AdvisoryFitFinding(
                requirementId="00000000-0000-4000-8000-000000000002",
                requirementKind="required",
                finding="SUPPORTED",
                confidence="HIGH",
                citations=["github:evidence-1"],
                uncertainty=[],
                explanation="The supplied evidence supports the requirement.",
            )
        ]
    )


def test_returns_no_assessable_evidence_without_calling_the_provider() -> None:
    calls = 0

    async def provider(_request: AdvisoryFitInput) -> AdvisoryFitProviderOutput:
        nonlocal calls
        calls += 1
        return completed_provider_output()

    request = AdvisoryFitInput.model_validate(
        {**assessment_request_payload(), "allowedEvidenceIds": []}
    )

    result = asyncio.run(generate_advisory_fit(request, provider=provider))

    assert result.status == "NOT_STARTED_NO_ASSESSABLE_EVIDENCE"
    assert result.findings == []
    assert calls == 0


def test_returns_completed_findings_and_safe_metadata() -> None:
    async def provider(_request: AdvisoryFitInput) -> AdvisoryFitProviderOutput:
        return completed_provider_output()

    request = AdvisoryFitInput.model_validate(assessment_request_payload())
    result = asyncio.run(generate_advisory_fit(request, provider=provider))

    assert result.status == "COMPLETED"
    assert result.findings[0].requirement_id == request.requirements[0].id
    assert result.metadata is not None
    assert result.metadata.provider == "deterministic-fake"
    assert result.metadata.prompt_version == "advisory-fit-v1"


def test_preserves_preferred_classification_confidence_and_uncertainty() -> None:
    payload = assessment_request_payload()
    payload["requirements"].append(
        {
            "id": "00000000-0000-4000-8000-000000000004",
            "kind": "preferred",
            "position": 1,
            "text": "GraphQL",
        }
    )
    request = AdvisoryFitInput.model_validate(payload)

    async def provider(_request: AdvisoryFitInput) -> AdvisoryFitProviderOutput:
        return AdvisoryFitProviderOutput(
            findings=[
                *completed_provider_output().findings,
                AdvisoryFitFinding(
                    requirementId="00000000-0000-4000-8000-000000000004",
                    requirementKind="preferred",
                    finding="INCONCLUSIVE",
                    confidence="LOW",
                    citations=["github:evidence-1"],
                    uncertainty=["The supplied evidence does not mention GraphQL."],
                    explanation="The authorized evidence is inconclusive for this preferred Requirement.",
                ),
            ]
        )

    result = asyncio.run(generate_advisory_fit(request, provider=provider))

    assert result.findings[1].requirement_kind == "preferred"
    assert result.findings[1].confidence == "LOW"
    assert result.findings[1].uncertainty


def test_rejects_provider_citations_outside_the_allowed_evidence_scope() -> None:
    async def provider(_request: AdvisoryFitInput) -> AdvisoryFitProviderOutput:
        output = completed_provider_output()
        output.findings[0].citations = ["invented-evidence"]
        return output

    request = AdvisoryFitInput.model_validate(assessment_request_payload())

    with pytest.raises(AdvisoryFitProviderError, match="citation"):
        asyncio.run(generate_advisory_fit(request, provider=provider))


def test_preserves_a_provider_system_limit_without_an_attempt() -> None:
    async def provider(_request: AdvisoryFitInput) -> AdvisoryFitProviderOutput:
        raise AdvisoryFitProviderSystemLimit("limit reached")

    request = AdvisoryFitInput.model_validate(assessment_request_payload())

    result = asyncio.run(generate_advisory_fit(request, provider=provider))

    assert result.status == "NOT_STARTED_SYSTEM_LIMIT"
    assert result.findings == []


def test_maps_gateway_rate_limits_to_system_limit(monkeypatch) -> None:
    request = httpx.Request("POST", "https://gateway.test/chat")
    response = httpx.Response(429, request=request)

    class LimitedGateway:
        async def generate_structured(self, **_kwargs):
            raise httpx.HTTPStatusError(
                "rate limited",
                request=request,
                response=response,
            )

    monkeypatch.setattr(
        "sharek_agents.common.llm.get_llm",
        lambda: LimitedGateway(),
    )

    request_data = AdvisoryFitInput.model_validate(assessment_request_payload())

    with pytest.raises(AdvisoryFitProviderSystemLimit):
        asyncio.run(_invoke_provider(request_data))


def test_default_provider_uses_the_strict_output_schema_and_truthful_metadata(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class StrictProvider:
        def invoke(self, _prompt: str):
            raise AssertionError("plain-text generation must not be used")

        async def generate_structured(
            self,
            *,
            system_prompt,
            user_prompt,
            response_model,
        ):
            captured["system_prompt"] = system_prompt
            captured["user_prompt"] = user_prompt
            captured["response_model"] = response_model
            return completed_provider_output()

    monkeypatch.setattr(
        "sharek_agents.common.llm.get_llm",
        lambda: StrictProvider(),
    )
    monkeypatch.setattr(
        "sharek_agents.common.llm.get_provider_metadata",
        lambda: ("openrouter", "openrouter/free"),
    )

    request_data = AdvisoryFitInput.model_validate(assessment_request_payload())
    response = asyncio.run(_invoke_provider(request_data))

    assert captured["response_model"] is AdvisoryFitProviderOutput
    assert "Return only one JSON object" in str(captured["system_prompt"])
    assert "ASSESSMENT REQUEST DATA" in str(captured["user_prompt"])
    assert response.output == completed_provider_output()
    assert response.provider == "openrouter"
    assert response.model == "openrouter/free"


def test_retries_one_transient_provider_failure(monkeypatch) -> None:
    calls = 0

    async def provider(_request: AdvisoryFitInput) -> AdvisoryFitProviderOutput:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise AdvisoryFitProviderTimeout("temporary timeout")
        return completed_provider_output()

    monkeypatch.setattr(
        "sharek_agents.config.settings.ai_advisory_fit_max_retries",
        1,
    )

    request = AdvisoryFitInput.model_validate(assessment_request_payload())
    result = asyncio.run(generate_advisory_fit(request, provider=provider))

    assert result.status == "COMPLETED"
    assert calls == 2


def test_fails_closed_after_the_bounded_retry_count(monkeypatch) -> None:
    calls = 0

    async def provider(_request: AdvisoryFitInput) -> AdvisoryFitProviderOutput:
        nonlocal calls
        calls += 1
        raise AdvisoryFitProviderError("provider unavailable")

    monkeypatch.setattr(
        "sharek_agents.config.settings.ai_advisory_fit_max_retries",
        1,
    )

    request = AdvisoryFitInput.model_validate(assessment_request_payload())

    with pytest.raises(AdvisoryFitProviderError):
        asyncio.run(generate_advisory_fit(request, provider=provider))

    assert calls == 2


def test_rejects_forbidden_aggregate_fit_output() -> None:
    payload = {
        "status": "COMPLETED",
        "findings": [completed_provider_output().findings[0].model_dump()],
        "metadata": {
            "provider": "test-provider",
            "model": "test-model",
            "promptVersion": "advisory-fit-v1",
            "schemaVersion": "advisory-fit-v1",
            "serviceVersion": "test-service",
        },
        "fitPercentage": 100,
    }

    with pytest.raises(ValidationError):
        AdvisoryFitResult.model_validate(payload)


def test_fails_closed_for_malformed_provider_response_metadata() -> None:
    async def provider(
        _request: AdvisoryFitInput,
    ) -> AdvisoryFitProviderResponse:
        return AdvisoryFitProviderResponse(
            output=completed_provider_output(),
            provider="",
            model="test-model",
        )

    request = AdvisoryFitInput.model_validate(assessment_request_payload())

    with pytest.raises(AdvisoryFitProviderError):
        asyncio.run(generate_advisory_fit(request, provider=provider))


def test_internal_endpoint_rejects_missing_credentials(monkeypatch) -> None:
    monkeypatch.setattr(
        "sharek_agents.security.settings.ai_service_auth_token",
        "internal-test-token-that-is-long-enough",
    )

    response = post_json(
        "/advisory-fit/assess",
        json=assessment_request_payload(),
    )

    assert response.status_code == 401


def test_internal_endpoint_returns_the_backend_advisory_fit_contract(
    monkeypatch,
) -> None:
    token = "internal-test-token-that-is-long-enough"
    expected = AdvisoryFitResult(
        status="COMPLETED",
        findings=completed_provider_output().findings,
        metadata=AdvisoryFitMetadata(
            provider="test-provider",
            model="test-model",
            promptVersion="test-prompt",
            schemaVersion="advisory-fit-v1",
            serviceVersion="test-service",
        ),
    )

    async def completed_assessment(
        _request: AdvisoryFitInput,
    ) -> AdvisoryFitResult:
        return expected

    monkeypatch.setattr(
        "sharek_agents.security.settings.ai_service_auth_token",
        token,
    )
    monkeypatch.setattr(
        "sharek_agents.main.analyze_advisory_fit",
        completed_assessment,
    )

    response = post_json(
        "/advisory-fit/assess",
        headers={"Authorization": f"Bearer {token}"},
        json=assessment_request_payload(),
    )

    assert response.status_code == 200
    assert response.json() == expected.model_dump(mode="json", by_alias=True)
