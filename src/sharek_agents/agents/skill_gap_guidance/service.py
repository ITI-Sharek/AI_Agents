from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx
from pydantic import ValidationError

from sharek_agents.agents.skill_gap_guidance.prompts import (
    SYSTEM_PROMPT,
    render_skill_gap_guidance_prompt,
)
from sharek_agents.agents.skill_gap_guidance.retrieval import (
    retrieve_curated_resources,
)
from sharek_agents.agents.skill_gap_guidance.schemas import (
    GuidanceMetadata,
    GuidanceProviderOutput,
    SkillGapGuidanceInput,
    SkillGapGuidanceResult,
)
from sharek_agents.config import settings


PROMPT_VERSION = "skill-gap-guidance-v1"
SCHEMA_VERSION = "skill-gap-guidance-v1"


class SkillGapGuidanceProviderError(Exception):
    """A provider response cannot be accepted as guidance."""


class SkillGapGuidanceProviderTimeout(SkillGapGuidanceProviderError):
    """The provider exceeded the bounded guidance timeout."""


class SkillGapGuidanceProviderSystemLimit(SkillGapGuidanceProviderError):
    """The provider or configured service limit prevented an attempt."""


@dataclass(frozen=True)
class SkillGapGuidanceProviderResponse:
    output: GuidanceProviderOutput
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


Provider = Callable[
    [SkillGapGuidanceInput],
    Awaitable[GuidanceProviderOutput | SkillGapGuidanceProviderResponse],
]


async def _invoke_with_retries(
    provider: Provider,
    input_data: SkillGapGuidanceInput,
) -> GuidanceProviderOutput | SkillGapGuidanceProviderResponse:
    max_retries = max(0, settings.ai_skill_gap_guidance_max_retries)
    for attempt in range(max_retries + 1):
        try:
            return await provider(input_data)
        except SkillGapGuidanceProviderSystemLimit:
            raise
        except (SkillGapGuidanceProviderTimeout, SkillGapGuidanceProviderError):
            if attempt >= max_retries:
                raise
    raise AssertionError("bounded provider retry loop did not return or raise")


async def _invoke_provider(
    input_data: SkillGapGuidanceInput,
) -> SkillGapGuidanceProviderResponse:
    from sharek_agents.common.llm import generate_structured, get_provider_metadata

    prompt = render_skill_gap_guidance_prompt(input_data)
    try:
        output = await asyncio.wait_for(
            generate_structured(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                response_model=GuidanceProviderOutput,
            ),
            timeout=settings.ai_skill_gap_guidance_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise SkillGapGuidanceProviderTimeout(
            "Skill-gap guidance provider timed out"
        ) from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in {402, 429}:
            raise SkillGapGuidanceProviderSystemLimit(
                "Skill-gap guidance provider system limit is active"
            ) from exc
        raise SkillGapGuidanceProviderError(
            "Skill-gap guidance provider failed"
        ) from exc
    except Exception as exc:
        raise SkillGapGuidanceProviderError(
            "Skill-gap guidance provider failed"
        ) from exc

    provider, model = get_provider_metadata()
    return SkillGapGuidanceProviderResponse(
        output=output,
        provider=provider,
        model=model,
    )


def _coerce_provider_response(
    response: GuidanceProviderOutput | SkillGapGuidanceProviderResponse,
) -> SkillGapGuidanceProviderResponse:
    if isinstance(response, SkillGapGuidanceProviderResponse):
        output = response.output
        if not isinstance(output, GuidanceProviderOutput):
            output = GuidanceProviderOutput.model_validate(output)
        return SkillGapGuidanceProviderResponse(
            output=output,
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
    if not isinstance(response, GuidanceProviderOutput):
        response = GuidanceProviderOutput.model_validate(response)
    return SkillGapGuidanceProviderResponse(
        output=response,
        provider="deterministic-fake",
        model="deterministic-fake",
    )


def _validate_input_scope(input_data: SkillGapGuidanceInput) -> set[str]:
    allowed = set(input_data.allowed_evidence_ids)
    if not allowed:
        return allowed
    supplied = {item.evidence_id for item in input_data.evidence}
    supplied.update(skill.evidence_id for skill in input_data.approved_skills)
    supplied.update(
        f"requirement:{requirement.id}" for requirement in input_data.requirements
    )
    if supplied != allowed:
        raise SkillGapGuidanceProviderError(
            "guidance input scope does not exactly match its evidence"
        )
    return allowed


def _validate_provider_scope(
    output: GuidanceProviderOutput,
    allowed_evidence_ids: set[str],
    allowed_resource_urls: set[str],
) -> None:
    cited_ids: set[str] = set()
    for item in output.missing_skills:
        cited_ids.update(item.evidence_ids)
    for item in output.recommended_technologies:
        cited_ids.update(item.evidence_ids)
    for item in output.learning_resources:
        cited_ids.update(item.evidence_ids)
    for item in output.practice_projects:
        cited_ids.update(item.evidence_ids)
    for item in output.improvement_path:
        cited_ids.update(item.evidence_ids)
    cited_ids.update(source.evidence_id for source in output.sources)
    if not cited_ids.issubset(allowed_evidence_ids):
        raise SkillGapGuidanceProviderError(
            "guidance output contains a citation outside the allowed scope"
        )
    if any(
        resource.url not in allowed_resource_urls
        for resource in output.learning_resources
    ):
        raise SkillGapGuidanceProviderError(
            "guidance output contains a resource outside the curated catalog"
        )


async def generate_skill_gap_guidance(
    input_data: SkillGapGuidanceInput,
    *,
    provider: Provider | None = None,
) -> SkillGapGuidanceResult:
    """Generate one explicit, bounded, source-attributed guidance result."""
    allowed_evidence_ids = _validate_input_scope(input_data)
    if not allowed_evidence_ids:
        return SkillGapGuidanceResult(status="NOT_STARTED_NO_ASSESSABLE_EVIDENCE")
    allowed_resource_urls = {
        resource.url for resource in retrieve_curated_resources(input_data)
    }

    started_at = time.perf_counter()
    try:
        response = _coerce_provider_response(
            await _invoke_with_retries(provider or _invoke_provider, input_data)
        )
    except SkillGapGuidanceProviderSystemLimit:
        return SkillGapGuidanceResult(status="NOT_STARTED_SYSTEM_LIMIT")
    except SkillGapGuidanceProviderTimeout:
        raise
    except SkillGapGuidanceProviderError:
        raise
    except asyncio.TimeoutError as exc:
        raise SkillGapGuidanceProviderTimeout(
            "Skill-gap guidance provider timed out"
        ) from exc
    except ValidationError as exc:
        raise SkillGapGuidanceProviderError(
            "Skill-gap guidance provider returned invalid output"
        ) from exc
    except Exception as exc:
        raise SkillGapGuidanceProviderError(
            "Skill-gap guidance provider failed"
        ) from exc

    try:
        _validate_provider_scope(
            response.output,
            allowed_evidence_ids,
            allowed_resource_urls,
        )
        latency_ms = max(0, round((time.perf_counter() - started_at) * 1000))
        metadata = GuidanceMetadata(
            provider=response.provider,
            model=response.model,
            promptVersion=PROMPT_VERSION,
            schemaVersion=SCHEMA_VERSION,
            serviceVersion=settings.service_version,
            latencyMs=latency_ms,
            inputTokens=response.input_tokens,
            outputTokens=response.output_tokens,
        )
    except SkillGapGuidanceProviderError:
        raise
    except ValidationError as exc:
        raise SkillGapGuidanceProviderError(
            "Skill-gap guidance provider returned invalid metadata"
        ) from exc

    return SkillGapGuidanceResult(
        status="COMPLETED",
        missingSkills=response.output.missing_skills,
        recommendedTechnologies=response.output.recommended_technologies,
        learningResources=response.output.learning_resources,
        practiceProjects=response.output.practice_projects,
        improvementPath=response.output.improvement_path,
        sources=response.output.sources,
        metadata=metadata,
    )
