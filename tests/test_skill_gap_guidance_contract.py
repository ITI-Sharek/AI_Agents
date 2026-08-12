from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from sharek_agents.agents.skill_gap_guidance.schemas import (
    GuidanceProviderOutput,
    SkillGapGuidanceInput,
)
from sharek_agents.agents.skill_gap_guidance.service import (
    SkillGapGuidanceProviderError,
    SkillGapGuidanceProviderSystemLimit,
    generate_skill_gap_guidance,
)


def payload() -> dict:
    return {
        "guidanceRequestId": "guidance-1",
        "requirements": [
            {"id": "req-1", "kind": "required", "position": 0, "text": "Learn Docker"}
        ],
        "approvedSkills": [],
        "evidence": [
            {
                "evidenceId": "requirement:req-1",
                "type": "contribution_requirement",
                "label": "Docker requirement",
                "summary": "The task requires Docker fundamentals.",
            }
        ],
        "allowedEvidenceIds": ["requirement:req-1"],
        "requestedAt": "2026-08-11T12:00:00.000Z",
        "contractVersion": "skill-gap-guidance-v1",
    }


def provider_output(*, citation: str = "requirement:req-1", url: str = "https://docs.docker.com/get-started/") -> GuidanceProviderOutput:
    return GuidanceProviderOutput.model_validate(
        {
            "missingSkills": [
                {
                    "requirementId": "req-1",
                    "skillName": "Docker",
                    "gap": "not_evidenced",
                    "explanation": "Docker is not evidenced in the supplied snapshot.",
                    "evidenceIds": [citation],
                    "uncertainty": [],
                }
            ],
            "recommendedTechnologies": [],
            "learningResources": [
                {
                    "title": "Docker Get Started",
                    "resourceType": "tutorial",
                    "url": url,
                    "rationale": "Practice container fundamentals.",
                    "evidenceIds": [citation],
                }
            ],
            "practiceProjects": [],
            "improvementPath": [],
            "sources": [
                {
                    "evidenceId": citation,
                    "label": "Docker requirement",
                    "type": "contribution_requirement",
                }
            ],
        }
    )


def test_contract_forbids_workflow_fields() -> None:
    request = SkillGapGuidanceInput.model_validate(payload())
    assert request.contract_version == "skill-gap-guidance-v1"
    with pytest.raises(ValidationError):
        SkillGapGuidanceInput.model_validate({**payload(), "applicationStatus": "rejected"})


def test_empty_scope_does_not_call_provider() -> None:
    calls = 0

    async def provider(_request: SkillGapGuidanceInput) -> GuidanceProviderOutput:
        nonlocal calls
        calls += 1
        return provider_output()

    request = SkillGapGuidanceInput.model_validate(
        {**payload(), "evidence": [], "allowedEvidenceIds": []}
    )
    result = asyncio.run(generate_skill_gap_guidance(request, provider=provider))
    assert result.status == "NOT_STARTED_NO_ASSESSABLE_EVIDENCE"
    assert calls == 0


def test_returns_source_scoped_guidance() -> None:
    async def provider(_request: SkillGapGuidanceInput) -> GuidanceProviderOutput:
        return provider_output()

    result = asyncio.run(
        generate_skill_gap_guidance(
            SkillGapGuidanceInput.model_validate(payload()), provider=provider
        )
    )
    body = result.model_dump(mode="json", by_alias=True)
    assert body["status"] == "COMPLETED"
    assert body["missingSkills"][0]["skillName"] == "Docker"
    assert body["metadata"]["provider"] == "deterministic-fake"
    assert {"score", "rank", "eligibility", "applicationStatus"}.isdisjoint(body)


def test_rejects_citations_outside_authorized_scope() -> None:
    async def provider(_request: SkillGapGuidanceInput) -> GuidanceProviderOutput:
        return provider_output(citation="private:evidence")

    with pytest.raises(SkillGapGuidanceProviderError):
        asyncio.run(
            generate_skill_gap_guidance(
                SkillGapGuidanceInput.model_validate(payload()), provider=provider
            )
        )


def test_rejects_learning_resources_outside_curated_catalog() -> None:
    async def provider(_request: SkillGapGuidanceInput) -> GuidanceProviderOutput:
        return provider_output(url="https://example.com/invented-course")

    with pytest.raises(SkillGapGuidanceProviderError):
        asyncio.run(
            generate_skill_gap_guidance(
                SkillGapGuidanceInput.model_validate(payload()), provider=provider
            )
        )


def test_system_limit_returns_safe_not_started_result() -> None:
    async def provider(_request: SkillGapGuidanceInput) -> GuidanceProviderOutput:
        raise SkillGapGuidanceProviderSystemLimit("quota")

    result = asyncio.run(
        generate_skill_gap_guidance(
            SkillGapGuidanceInput.model_validate(payload()), provider=provider
        )
    )
    assert result.status == "NOT_STARTED_SYSTEM_LIMIT"
    assert result.metadata is None
