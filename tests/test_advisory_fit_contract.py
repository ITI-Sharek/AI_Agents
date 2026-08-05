from __future__ import annotations

import asyncio
import os

import httpx
import pytest
from pydantic import ValidationError

from sharek_agents.agents.advisory_fit.schemas import (
    AdvisoryFitFinding,
    AdvisoryFitInput,
    AdvisoryFitProviderOutput,
)
from sharek_agents.agents.advisory_fit.service import (
    AdvisoryFitProviderError,
    AdvisoryFitProviderSystemLimit,
    generate_advisory_fit,
)
from sharek_agents.main import app


def payload() -> dict:
    return {
        "assessmentRequestId": "assessment-1",
        "requirements": [
            {"id": "req-1", "kind": "required", "position": 0, "text": "NestJS"},
            {"id": "req-2", "kind": "preferred", "position": 1, "text": "Redis"},
        ],
        "evidence": [{"evidenceId": "github:1", "privateMarker": "PRIVATE"}],
        "allowedEvidenceIds": ["github:1"],
        "requestedAt": "2026-08-05T12:00:00.000Z",
        "contractVersion": "advisory-fit-v1",
    }


def output() -> AdvisoryFitProviderOutput:
    return AdvisoryFitProviderOutput(
        findings=[
            AdvisoryFitFinding(
                requirementId="req-1",
                requirementKind="required",
                finding="SUPPORTED",
                confidence="HIGH",
                citations=["github:1"],
                uncertainty=[],
                explanation="The fixed evidence supports this Requirement.",
            ),
            AdvisoryFitFinding(
                requirementId="req-2",
                requirementKind="preferred",
                finding="INCONCLUSIVE",
                confidence="LOW",
                citations=["github:1"],
                uncertainty=["The evidence is limited."],
                explanation="The fixed evidence is inconclusive.",
            ),
        ]
    )


def test_accepts_exact_backend_contract_and_forbids_extra_fields() -> None:
    request = AdvisoryFitInput.model_validate(payload())
    assert request.requirements[1].kind == "preferred"
    with pytest.raises(ValidationError):
        AdvisoryFitInput.model_validate({**payload(), "applicationStatus": "ACCEPTED"})


def test_no_evidence_returns_not_started_without_provider_call() -> None:
    calls = 0

    async def provider(_request: AdvisoryFitInput) -> AdvisoryFitProviderOutput:
        nonlocal calls
        calls += 1
        return output()

    request = AdvisoryFitInput.model_validate({**payload(), "allowedEvidenceIds": []})
    result = asyncio.run(generate_advisory_fit(request, provider=provider))
    assert result.status == "NOT_STARTED_NO_ASSESSABLE_EVIDENCE"
    assert calls == 0


def test_returns_complete_decision_neutral_findings() -> None:
    async def provider(_request: AdvisoryFitInput) -> AdvisoryFitProviderOutput:
        return output()

    result = asyncio.run(
        generate_advisory_fit(AdvisoryFitInput.model_validate(payload()), provider=provider)
    )
    body = result.model_dump(mode="json", by_alias=True)
    assert body["status"] == "COMPLETED"
    assert [item["requirementId"] for item in body["findings"]] == ["req-1", "req-2"]
    prohibited = {"score", "rank", "eligibility", "recommendation", "applicationStatus"}
    assert prohibited.isdisjoint(body)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value.findings.pop(), "cover"),
        (lambda value: setattr(value.findings[0], "requirement_kind", "preferred"), "classification"),
        (lambda value: setattr(value.findings[0], "citations", ["github:private"]), "citation"),
    ],
)
def test_rejects_incomplete_mismatched_or_unauthorized_output(mutate, match) -> None:
    async def provider(_request: AdvisoryFitInput) -> AdvisoryFitProviderOutput:
        value = output()
        mutate(value)
        return value

    with pytest.raises(AdvisoryFitProviderError, match=match):
        asyncio.run(
            generate_advisory_fit(
                AdvisoryFitInput.model_validate(payload()), provider=provider
            )
        )


def test_system_limit_is_not_a_negative_finding() -> None:
    async def provider(_request: AdvisoryFitInput) -> AdvisoryFitProviderOutput:
        raise AdvisoryFitProviderSystemLimit("limited")

    result = asyncio.run(
        generate_advisory_fit(AdvisoryFitInput.model_validate(payload()), provider=provider)
    )
    assert result.status == "NOT_STARTED_SYSTEM_LIMIT"
    assert result.findings == []


def test_retries_once_and_then_succeeds(monkeypatch) -> None:
    monkeypatch.setenv("AI_ADVISORY_FIT_MAX_RETRIES", "1")
    calls = 0

    async def provider(_request: AdvisoryFitInput) -> AdvisoryFitProviderOutput:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise AdvisoryFitProviderError("transient")
        return output()

    asyncio.run(
        generate_advisory_fit(AdvisoryFitInput.model_validate(payload()), provider=provider)
    )
    assert calls == 2


def post(path: str, *, token: str | None, body: dict):
    async def request():
        transport = httpx.ASGITransport(app=app)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(path, headers=headers, json=body)

    return asyncio.run(request())


def test_endpoint_requires_the_shared_service_token(monkeypatch) -> None:
    monkeypatch.setenv("AI_SERVICE_AUTH_TOKEN", "service-secret")
    assert post("/advisory-fit/assess", token=None, body=payload()).status_code == 401
    assert post("/advisory-fit/assess", token="wrong", body=payload()).status_code == 401


def test_endpoint_reports_missing_server_auth_configuration(monkeypatch) -> None:
    monkeypatch.delenv("AI_SERVICE_AUTH_TOKEN", raising=False)
    response = post("/advisory-fit/assess", token="anything", body=payload())
    assert response.status_code == 503
    assert "PRIVATE" not in response.text


def test_endpoint_validates_before_any_provider_call(monkeypatch) -> None:
    monkeypatch.setenv("AI_SERVICE_AUTH_TOKEN", "service-secret")
    invalid = payload()
    invalid["contractVersion"] = "eligibility-v1"
    response = post("/advisory-fit/assess", token="service-secret", body=invalid)
    assert response.status_code == 422
    assert "PRIVATE" not in response.text
