import asyncio

import httpx
import pytest
from pydantic import ValidationError

from sharek_agents.agents.skill_gap_guidance.schemas import (
    GuidanceProviderOutput,
    SkillGapGuidanceInput,
    SkillGapGuidanceResult,
)
from sharek_agents.agents.skill_gap_guidance.service import generate_skill_gap_guidance
from sharek_agents.agents.skill_gap_guidance.service import (
    SkillGapGuidanceProviderError,
)
from sharek_agents.main import app


def guidance_request_payload() -> dict:
    return {
        "guidanceRequestId": "guidance-request-1",
        "requirements": [
            {
                "id": "requirement-1",
                "kind": "required",
                "position": 0,
                "text": "Build a scheduled data pipeline with Apache Airflow",
            }
        ],
        "approvedSkills": [
            {
                "evidenceId": "skill:python",
                "name": "Python",
                "proficiency": "advanced",
                "evidenceSummary": "Approved skill from reviewed profile",
            }
        ],
        "evidence": [
            {
                "evidenceId": "requirement:requirement-1",
                "type": "contribution_requirement",
                "label": "Contribution Request requirement",
                "summary": "The request requires Airflow scheduling",
            },
            {
                "evidenceId": "skill:python",
                "type": "approved_skill",
                "label": "Python",
                "summary": "Approved skill from reviewed profile",
            },
        ],
        "allowedEvidenceIds": ["requirement:requirement-1", "skill:python"],
        "requestedAt": "2026-08-11T12:00:00Z",
        "contractVersion": "skill-gap-guidance-v1",
    }


def provider_output() -> GuidanceProviderOutput:
    return GuidanceProviderOutput.model_validate(
        {
            "missingSkills": [
                {
                    "requirementId": "requirement-1",
                    "skillName": "Apache Airflow",
                    "gap": "not_evidenced",
                    "explanation": "The supplied approved profile does not evidence Airflow.",
                    "evidenceIds": ["requirement:requirement-1"],
                    "uncertainty": [],
                }
            ],
            "recommendedTechnologies": [
                {
                    "name": "Apache Airflow",
                    "rationale": "It is named by the target requirement.",
                    "evidenceIds": ["requirement:requirement-1"],
                }
            ],
            "learningResources": [
                {
                    "title": "Apache Airflow documentation",
                    "resourceType": "documentation",
                    "url": "https://airflow.apache.org/docs/",
                    "rationale": "The official documentation is a direct source.",
                    "evidenceIds": ["requirement:requirement-1"],
                }
            ],
            "practiceProjects": [
                {
                    "title": "Build a scheduled ETL pipeline",
                    "description": "Create an Airflow DAG that loads a small dataset.",
                    "technologies": ["Apache Airflow", "Python"],
                    "evidenceIds": ["requirement:requirement-1", "skill:python"],
                }
            ],
            "improvementPath": [],
            "sources": [
                {
                    "evidenceId": "requirement:requirement-1",
                    "label": "Contribution Request requirement",
                    "type": "contribution_requirement",
                }
            ],
        }
    )


def test_guidance_returns_structured_recommendations_with_allowed_sources() -> None:
    request = SkillGapGuidanceInput.model_validate(guidance_request_payload())

    result = asyncio.run(
        generate_skill_gap_guidance(request, provider=lambda _request: _async_output())
    )

    assert isinstance(result, SkillGapGuidanceResult)
    assert result.status == "COMPLETED"
    assert result.missing_skills[0].skill_name == "Apache Airflow"
    assert result.learning_resources[0].url == "https://airflow.apache.org/docs/"


def test_empty_allowed_scope_does_not_call_the_provider() -> None:
    request_payload = guidance_request_payload()
    request_payload["allowedEvidenceIds"] = []
    request_payload["evidence"] = []
    request = SkillGapGuidanceInput.model_validate(request_payload)
    calls = 0

    async def provider(_request):
        nonlocal calls
        calls += 1
        return provider_output()

    result = asyncio.run(generate_skill_gap_guidance(request, provider=provider))

    assert result.status == "NOT_STARTED_NO_ASSESSABLE_EVIDENCE"
    assert calls == 0


def test_provider_cannot_return_a_learning_url_outside_curated_retrieval() -> None:
    payload = provider_output().model_dump(mode="json", by_alias=True)
    payload["learningResources"][0]["url"] = "https://example.com/unsafe"
    request = SkillGapGuidanceInput.model_validate(guidance_request_payload())

    with pytest.raises(SkillGapGuidanceProviderError):
        asyncio.run(
            generate_skill_gap_guidance(
                request,
                provider=lambda _request: _async_output(
                    GuidanceProviderOutput.model_validate(payload)
                ),
            )
        )


def test_result_schema_rejects_forbidden_decision_fields() -> None:
    with pytest.raises(ValidationError):
        SkillGapGuidanceResult.model_validate(
            {
                "status": "COMPLETED",
                "eligible": False,
            }
        )


def test_authenticated_stream_endpoint_emits_one_atomic_completed_event(monkeypatch) -> None:
    token = "internal-test-token-that-is-long-enough"

    async def completed(_request: SkillGapGuidanceInput) -> SkillGapGuidanceResult:
        return SkillGapGuidanceResult(
            status="COMPLETED",
            sources=[
                {
                    "evidenceId": "requirement:requirement-1",
                    "label": "Contribution Request requirement",
                    "type": "contribution_requirement",
                }
            ],
            metadata={
                "provider": "fixture",
                "model": "fixture",
                "promptVersion": "skill-gap-guidance-v1",
                "schemaVersion": "skill-gap-guidance-v1",
                "serviceVersion": "test",
            },
        )

    monkeypatch.setenv("AI_SERVICE_AUTH_TOKEN", token)
    monkeypatch.setattr(
        "sharek_agents.agents.skill_gap_guidance.endpoint.generate_skill_gap_guidance",
        completed,
    )

    response = asyncio.run(
        _post_stream(
            headers={"Authorization": f"Bearer {token}"},
            json=guidance_request_payload(),
        )
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "guidance.completed" in response.text


def test_authenticated_generate_endpoint_returns_structured_json(monkeypatch) -> None:
    token = "internal-test-token-that-is-long-enough"

    async def completed(_request: SkillGapGuidanceInput) -> SkillGapGuidanceResult:
        return SkillGapGuidanceResult(
            status="COMPLETED",
            sources=[
                {
                    "evidenceId": "requirement:requirement-1",
                    "label": "Contribution Request requirement",
                    "type": "contribution_requirement",
                }
            ],
            metadata={
                "provider": "fixture",
                "model": "fixture",
                "promptVersion": "skill-gap-guidance-v1",
                "schemaVersion": "skill-gap-guidance-v1",
                "serviceVersion": "test",
            },
        )

    monkeypatch.setenv("AI_SERVICE_AUTH_TOKEN", token)
    monkeypatch.setattr(
        "sharek_agents.agents.skill_gap_guidance.endpoint.generate_skill_gap_guidance",
        completed,
    )

    response = asyncio.run(
        _post(
            headers={"Authorization": f"Bearer {token}"},
            json=guidance_request_payload(),
        )
    )

    assert response.status_code == 200
    assert response.json()["status"] == "COMPLETED"


async def _post_stream(*, headers: dict[str, str], json: dict):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.post(
            "/skill-gap-guidance/stream",
            headers=headers,
            json=json,
        )


async def _post(*, headers: dict[str, str], json: dict):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.post(
            "/skill-gap-guidance/generate",
            headers=headers,
            json=json,
        )


async def _async_output(
    output: GuidanceProviderOutput | None = None,
) -> GuidanceProviderOutput:
    return output or provider_output()
