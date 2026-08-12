from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from sharek_agents.agents.contributor_matching.schemas import (
    ContributorMatchingInput,
    ContributorMatchingProviderOutput,
)
from sharek_agents.agents.contributor_matching.service import (
    ContributorMatchingProviderError,
    ContributorMatchingProviderSystemLimit,
    generate_contributor_matching,
)


def payload() -> dict:
    return {
        "matchingRequestId": "match-1",
        "contributionRequestId": "request-1",
        "title": "Add a NestJS API",
        "description": "Build an authenticated endpoint.",
        "requirements": [
            {"id": "req-1", "kind": "required", "position": 0, "text": "NestJS"}
        ],
        "candidates": [
            {
                "contributorId": "contributor-1",
                "displayName": "Contributor One",
                "username": "contributor-one",
                "approvedSkills": [
                    {
                        "skillProfileId": "skill-1",
                        "name": "NestJS",
                        "proficiency": "advanced",
                        "confidence": 0.92,
                        "evidenceIds": ["skill:evidence-1"],
                        "evidenceSummary": "Approved repository evidence.",
                    }
                ],
                "reputation": {
                    "rating": 4.8,
                    "completedContributions": 5,
                    "successRate": 90,
                    "topVerifiedSkills": ["NestJS"],
                },
            }
        ],
        "evidence": [
            {
                "evidenceId": "skill:evidence-1",
                "type": "approved_skill",
                "label": "Approved NestJS skill",
                "summary": "Repository-backed evidence.",
                "contributorId": "contributor-1",
            }
        ],
        "allowedEvidenceIds": ["skill:evidence-1"],
        "requestedAt": "2026-08-11T12:00:00.000Z",
        "contractVersion": "contributor-matching-v1",
    }


def provider_output(
    *, contributor_id: str = "contributor-1", skill_name: str = "NestJS", citation: str = "skill:evidence-1"
) -> ContributorMatchingProviderOutput:
    return ContributorMatchingProviderOutput.model_validate(
        {
            "matches": [
                {
                    "contributorId": contributor_id,
                    "matchScore": 0.91,
                    "confidence": "HIGH",
                    "justification": "Approved NestJS evidence aligns with the request.",
                    "matchedSkills": [
                        {
                            "name": skill_name,
                            "proficiency": "advanced",
                            "evidenceIds": [citation],
                        }
                    ],
                    "evidenceIds": [citation],
                }
            ]
        }
    )


def test_contract_forbids_owner_decision_fields() -> None:
    request = ContributorMatchingInput.model_validate(payload())
    assert request.candidates[0].contributor_id == "contributor-1"
    with pytest.raises(ValidationError):
        ContributorMatchingInput.model_validate({**payload(), "selectedContributorId": "contributor-1"})


def test_no_candidates_skips_provider() -> None:
    calls = 0

    async def provider(_request: ContributorMatchingInput) -> ContributorMatchingProviderOutput:
        nonlocal calls
        calls += 1
        return provider_output()

    request = ContributorMatchingInput.model_validate({**payload(), "candidates": []})
    result = asyncio.run(generate_contributor_matching(request, provider=provider))
    assert result.status == "NOT_STARTED_NO_CANDIDATES"
    assert calls == 0


def test_returns_evidence_scoped_recommendation() -> None:
    async def provider(_request: ContributorMatchingInput) -> ContributorMatchingProviderOutput:
        return provider_output()

    result = asyncio.run(
        generate_contributor_matching(
            ContributorMatchingInput.model_validate(payload()), provider=provider
        )
    )
    body = result.model_dump(mode="json", by_alias=True)
    assert body["status"] == "COMPLETED"
    assert body["matches"][0]["confidence"] == "HIGH"
    assert body["metadata"]["provider"] == "deterministic-fake"
    assert {"eligible", "selected", "applicationStatus"}.isdisjoint(body)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"contributor_id": "unknown"}, "unknown contributor"),
        ({"skill_name": "Private Skill"}, "approved candidate snapshot"),
        ({"citation": "private:evidence"}, "allowed scope"),
    ],
)
def test_rejects_output_outside_authorized_snapshot(kwargs: dict, message: str) -> None:
    async def provider(_request: ContributorMatchingInput) -> ContributorMatchingProviderOutput:
        return provider_output(**kwargs)

    with pytest.raises(ContributorMatchingProviderError, match=message):
        asyncio.run(
            generate_contributor_matching(
                ContributorMatchingInput.model_validate(payload()), provider=provider
            )
        )


def test_system_limit_returns_safe_not_started_result() -> None:
    async def provider(_request: ContributorMatchingInput) -> ContributorMatchingProviderOutput:
        raise ContributorMatchingProviderSystemLimit("quota")

    result = asyncio.run(
        generate_contributor_matching(
            ContributorMatchingInput.model_validate(payload()), provider=provider
        )
    )
    assert result.status == "NOT_STARTED_SYSTEM_LIMIT"
    assert result.matches == []
