from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx
from pydantic import ValidationError

from sharek_agents.agents.contributor_matching.prompts import (
    SYSTEM_PROMPT,
    render_contributor_matching_prompt,
)
from sharek_agents.agents.contributor_matching.schemas import (
    ContributorMatchingInput,
    ContributorMatchingMetadata,
    ContributorMatchingProviderMatch,
    ContributorMatchingProviderOutput,
    ContributorMatchingResult,
)
from sharek_agents.config import settings


PROMPT_VERSION = "contributor-matching-v1"
SCHEMA_VERSION = "contributor-matching-v1"


class ContributorMatchingProviderError(Exception):
    """A provider response cannot be accepted as a matching recommendation."""


class ContributorMatchingProviderTimeout(ContributorMatchingProviderError):
    """The provider exceeded the bounded matching timeout."""


class ContributorMatchingProviderSystemLimit(ContributorMatchingProviderError):
    """The provider or configured service limit prevented an attempt."""


@dataclass(frozen=True)
class ContributorMatchingProviderResponse:
    output: ContributorMatchingProviderOutput
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


Provider = Callable[
    [ContributorMatchingInput],
    Awaitable[ContributorMatchingProviderOutput | ContributorMatchingProviderResponse],
]


async def _invoke_with_retries(
    provider: Provider,
    input_data: ContributorMatchingInput,
) -> ContributorMatchingProviderOutput | ContributorMatchingProviderResponse:
    max_retries = max(0, settings.ai_contributor_matching_max_retries)
    for attempt in range(max_retries + 1):
        try:
            return await provider(input_data)
        except ContributorMatchingProviderSystemLimit:
            raise
        except (ContributorMatchingProviderTimeout, ContributorMatchingProviderError):
            if attempt >= max_retries:
                raise
    raise AssertionError("bounded provider retry loop did not return or raise")


async def _invoke_provider(input_data: ContributorMatchingInput) -> ContributorMatchingProviderResponse:
    from sharek_agents.common.llm import generate_structured, get_provider_metadata

    try:
        output = await asyncio.wait_for(
            generate_structured(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=render_contributor_matching_prompt(input_data),
                response_model=ContributorMatchingProviderOutput,
            ),
            timeout=settings.ai_contributor_matching_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise ContributorMatchingProviderTimeout(
            "Contributor matching provider timed out"
        ) from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in {402, 429}:
            raise ContributorMatchingProviderSystemLimit(
                "Contributor matching provider system limit is active"
            ) from exc
        raise ContributorMatchingProviderError("Contributor matching provider failed") from exc
    except Exception as exc:
        raise ContributorMatchingProviderError("Contributor matching provider failed") from exc

    provider, model = get_provider_metadata()
    return ContributorMatchingProviderResponse(
        output=output,
        provider=provider,
        model=model,
    )

def _coerce_provider_response(
    response: ContributorMatchingProviderOutput | ContributorMatchingProviderResponse,
) -> ContributorMatchingProviderResponse:
    if isinstance(response, ContributorMatchingProviderResponse):
        output = response.output
        if not isinstance(output, ContributorMatchingProviderOutput):
            output = ContributorMatchingProviderOutput.model_validate(output)
        return ContributorMatchingProviderResponse(
            output=output,
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
    if not isinstance(response, ContributorMatchingProviderOutput):
        response = ContributorMatchingProviderOutput.model_validate(response)
    return ContributorMatchingProviderResponse(
        output=response,
        provider="deterministic-fake",
        model="deterministic-fake",
    )


def _validate_provider_scope(
    output: ContributorMatchingProviderOutput,
    input_data: ContributorMatchingInput,
) -> None:
    candidates = {candidate.contributor_id: candidate for candidate in input_data.candidates}
    allowed_evidence_ids = set(input_data.allowed_evidence_ids)
    seen: set[str] = set()
    for match in output.matches:
        if match.contributor_id not in candidates:
            raise ContributorMatchingProviderError(
                "matching output contains an unknown contributor"
            )
        if match.contributor_id in seen:
            raise ContributorMatchingProviderError(
                "matching output contains a duplicate contributor"
            )
        seen.add(match.contributor_id)
        if any(evidence_id not in allowed_evidence_ids for evidence_id in match.evidence_ids):
            raise ContributorMatchingProviderError(
                "matching output contains an evidence citation outside the allowed scope"
            )
        candidate_skill_names = {
            skill.name.casefold() for skill in candidates[match.contributor_id].approved_skills
        }
        for skill in match.matched_skills:
            if skill.name.casefold() not in candidate_skill_names:
                raise ContributorMatchingProviderError(
                    "matching output contains a skill outside the approved candidate snapshot"
                )
            if any(evidence_id not in allowed_evidence_ids for evidence_id in skill.evidence_ids):
                raise ContributorMatchingProviderError(
                    "matched skill contains an evidence citation outside the allowed scope"
                )


async def generate_contributor_matching(
    input_data: ContributorMatchingInput,
    *,
    provider: Provider | None = None,
) -> ContributorMatchingResult:
    """Run one bounded, evidence-scoped contributor matching recommendation."""
    if not input_data.candidates:
        return ContributorMatchingResult(status="NOT_STARTED_NO_CANDIDATES")

    started_at = time.perf_counter()
    try:
        response = _coerce_provider_response(
            await _invoke_with_retries(provider or _invoke_provider, input_data)
        )
    except ContributorMatchingProviderSystemLimit:
        return ContributorMatchingResult(status="NOT_STARTED_SYSTEM_LIMIT")
    except (ContributorMatchingProviderTimeout, ContributorMatchingProviderError):
        raise
    except ValidationError as exc:
        raise ContributorMatchingProviderError(
            "Contributor matching provider returned invalid output"
        ) from exc
    except Exception as exc:
        raise ContributorMatchingProviderError("Contributor matching provider failed") from exc

    _validate_provider_scope(response.output, input_data)
    try:
        metadata = ContributorMatchingMetadata(
            provider=response.provider,
            model=response.model,
            promptVersion=PROMPT_VERSION,
            schemaVersion=SCHEMA_VERSION,
            serviceVersion=settings.service_version,
            latencyMs=max(0, round((time.perf_counter() - started_at) * 1000)),
            inputTokens=response.input_tokens,
            outputTokens=response.output_tokens,
        )
    except ValidationError as exc:
        raise ContributorMatchingProviderError(
            "Contributor matching provider returned invalid metadata"
        ) from exc
    return ContributorMatchingResult(
        status="COMPLETED",
        matches=response.output.matches,
        metadata=metadata,
    )
