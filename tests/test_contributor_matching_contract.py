import asyncio

import httpx
import pytest
from pydantic import ValidationError

from sharek_agents.agents.contributor_matching.schemas import (
    ContributorMatchingInput,
    ContributorMatchingProviderOutput,
    ContributorMatchingResult,
)
from sharek_agents.agents.contributor_matching.service import (
    ContributorMatchingProviderError,
    generate_contributor_matching,
)
from sharek_agents.main import app


def matching_request_payload() -> dict:
    return {
        "matchingRequestId": "match-request-1",
        "contributionRequestId": "request-1",
        "title": "Add JWT Authentication",
        "description": "Implement secure JWT authentication for the API.",
        "requirements": [
            {
                "id": "requirement-1",
                "kind": "required",
                "position": 0,
                "text": "Node.js and JWT authentication",
            }
        ],
        "candidates": [
            {
                "contributorId": "contributor-1",
                "displayName": "Sara Ahmed",
                "username": "sara-dev",
                "approvedSkills": [
                    {
                        "skillProfileId": "skill-1",
                        "name": "Node.js",
                        "proficiency": "advanced",
                        "confidence": 0.94,
                        "evidenceIds": ["github:sara/api"],
                    }
                ],
                "reputation": {
                    "rating": 4.7,
                    "completedContributions": 13,
                    "successRate": 93.0,
                    "topVerifiedSkills": ["Node.js"],
                },
            }
        ],
        "evidence": [
            {
                "evidenceId": "requirement:requirement-1",
                "type": "contribution_requirement",
                "label": "Required Requirement",
                "summary": "Node.js and JWT authentication",
            },
            {
                "evidenceId": "github:sara/api",
                "type": "retrieved_evidence",
                "label": "Sara's approved Node.js evidence",
                "summary": "Approved skill evidence from reviewed repository activity",
                "contributorId": "contributor-1",
            },
        ],
        "allowedEvidenceIds": ["requirement:requirement-1", "github:sara/api"],
        "requestedAt": "2026-08-11T12:00:00Z",
        "contractVersion": "contributor-matching-v1",
    }


def provider_output() -> ContributorMatchingProviderOutput:
    return ContributorMatchingProviderOutput.model_validate(
        {
            "matches": [
                {
                    "contributorId": "contributor-1",
                    "matchScore": 0.94,
                    "confidence": "HIGH",
                    "justification": "Strong approved Node.js evidence and verified delivery history.",
                    "matchedSkills": [
                        {
                            "name": "Node.js",
                            "proficiency": "advanced",
                            "evidenceIds": ["github:sara/api"],
                        }
                    ],
                    "evidenceIds": ["requirement:requirement-1", "github:sara/api"],
                }
            ]
        }
    )


def test_matching_returns_rankable_structured_recommendations() -> None:
    request = ContributorMatchingInput.model_validate(matching_request_payload())

    result = asyncio.run(
        generate_contributor_matching(request, provider=lambda _request: _async_output())
    )

    assert isinstance(result, ContributorMatchingResult)
    assert result.status == "COMPLETED"
    assert result.matches[0].contributor_id == "contributor-1"
    assert result.matches[0].confidence == "HIGH"


def test_provider_cannot_return_unknown_candidate_or_evidence() -> None:
    payload = provider_output().model_dump(mode="json", by_alias=True)
    payload["matches"][0]["contributorId"] = "not-a-candidate"
    request = ContributorMatchingInput.model_validate(matching_request_payload())

    with pytest.raises(ContributorMatchingProviderError):
        asyncio.run(
            generate_contributor_matching(
                request,
                provider=lambda _request: _async_output(
                    ContributorMatchingProviderOutput.model_validate(payload)
                ),
            )
        )


def test_empty_candidate_scope_does_not_call_provider() -> None:
    payload = matching_request_payload()
    payload["candidates"] = []
    request = ContributorMatchingInput.model_validate(payload)
    calls = 0

    async def provider(_request):
        nonlocal calls
        calls += 1
        return provider_output()

    result = asyncio.run(generate_contributor_matching(request, provider=provider))

    assert result.status == "NOT_STARTED_NO_CANDIDATES"
    assert calls == 0


def test_result_schema_rejects_decision_fields() -> None:
    with pytest.raises(ValidationError):
        ContributorMatchingResult.model_validate(
            {"status": "COMPLETED", "eligible": True}
        )


def test_authenticated_endpoint_returns_the_matching_contract(monkeypatch) -> None:
    token = "internal-test-token-that-is-long-enough"

    async def completed(_request: ContributorMatchingInput) -> ContributorMatchingResult:
        return ContributorMatchingResult(
            status="COMPLETED",
            matches=[provider_output().matches[0]],
            metadata={
                "provider": "fixture",
                "model": "fixture",
                "promptVersion": "contributor-matching-v1",
                "schemaVersion": "contributor-matching-v1",
                "serviceVersion": "test",
            },
        )

    monkeypatch.setenv("AI_SERVICE_AUTH_TOKEN", token)
    monkeypatch.setattr(
        "sharek_agents.agents.contributor_matching.endpoint.generate_contributor_matching",
        completed,
    )

    response = asyncio.run(_post(
        headers={"Authorization": f"Bearer {token}"},
        json=matching_request_payload(),
    ))

    assert response.status_code == 200
    assert response.json()["status"] == "COMPLETED"


async def _post(*, headers: dict[str, str], json: dict):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.post(
            "/contributor-matching/generate",
            headers=headers,
            json=json,
        )


async def _async_output(
    output: ContributorMatchingProviderOutput | None = None,
) -> ContributorMatchingProviderOutput:
    return output or provider_output()
