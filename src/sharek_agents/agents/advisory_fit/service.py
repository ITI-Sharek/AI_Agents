from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx
from pydantic import ValidationError

from sharek_agents.agents.advisory_fit.prompts import (
    SYSTEM_PROMPT,
    render_advisory_fit_prompt,
)
from sharek_agents.agents.advisory_fit.schemas import (
    AdvisoryFitInput,
    AdvisoryFitMetadata,
    AdvisoryFitProviderOutput,
    AdvisoryFitResult,
)
from sharek_agents.config import settings


PROMPT_VERSION = "advisory-fit-v1"
SCHEMA_VERSION = "advisory-fit-v1"


class AdvisoryFitProviderError(Exception):
    """A provider response cannot be accepted as a valid assessment."""


class AdvisoryFitProviderTimeout(AdvisoryFitProviderError):
    """The provider exceeded the bounded assessment timeout."""


class AdvisoryFitProviderSystemLimit(AdvisoryFitProviderError):
    """The provider or configured service limit prevented an attempt."""


@dataclass(frozen=True)
class AdvisoryFitProviderResponse:
    output: AdvisoryFitProviderOutput
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


Provider = Callable[
    [AdvisoryFitInput],
    Awaitable[AdvisoryFitProviderOutput | AdvisoryFitProviderResponse],
]


async def _invoke_with_retries(
    provider: Provider,
    input_data: AdvisoryFitInput,
) -> AdvisoryFitProviderOutput | AdvisoryFitProviderResponse:
    max_retries = max(0, settings.ai_advisory_fit_max_retries)
    for attempt in range(max_retries + 1):
        try:
            return await provider(input_data)
        except AdvisoryFitProviderSystemLimit:
            raise
        except (AdvisoryFitProviderTimeout, AdvisoryFitProviderError):
            if attempt >= max_retries:
                raise

    raise AssertionError("bounded provider retry loop did not return or raise")


async def _invoke_provider(
    input_data: AdvisoryFitInput,
) -> AdvisoryFitProviderResponse:
    from sharek_agents.common.llm import generate_structured, get_provider_metadata

    prompt = render_advisory_fit_prompt(input_data)

    try:
        output = await asyncio.wait_for(
            generate_structured(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                response_model=AdvisoryFitProviderOutput,
            ),
            timeout=settings.ai_skill_profile_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise AdvisoryFitProviderTimeout(
            "Advisory Fit provider timed out"
        ) from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in {402, 429}:
            raise AdvisoryFitProviderSystemLimit(
                "Advisory Fit provider system limit is active"
            ) from exc
        raise AdvisoryFitProviderError("Advisory Fit provider failed") from exc
    except Exception as exc:
        raise AdvisoryFitProviderError("Advisory Fit provider failed") from exc

    provider, model = get_provider_metadata()

    # The configured adapters do not currently expose token counts. Latency is
    # measured locally and persisted by NestJS as safe technical metadata.
    return AdvisoryFitProviderResponse(
        output=output,
        provider=provider,
        model=model,
    )


def _validate_provider_coverage(
    output: AdvisoryFitProviderOutput,
    input_data: AdvisoryFitInput,
) -> None:
    requirements = {item.id: item.kind for item in input_data.requirements}
    findings = output.findings
    finding_ids = [finding.requirement_id for finding in findings]

    if len(findings) != len(requirements) or set(finding_ids) != set(requirements):
        raise AdvisoryFitProviderError(
            "AI findings must cover each Requirement exactly once"
        )
    if len(finding_ids) != len(set(finding_ids)):
        raise AdvisoryFitProviderError(
            "AI findings must not contain duplicate Requirements"
        )

    allowed_evidence_ids = set(input_data.allowed_evidence_ids)
    for finding in findings:
        if finding.requirement_kind != requirements[finding.requirement_id]:
            raise AdvisoryFitProviderError(
                "AI finding Requirement classification does not match the snapshot"
            )
        if any(citation not in allowed_evidence_ids for citation in finding.citations):
            raise AdvisoryFitProviderError(
                "AI finding contains a citation outside the allowed evidence scope"
            )


def _coerce_provider_response(
    response: AdvisoryFitProviderOutput | AdvisoryFitProviderResponse,
) -> AdvisoryFitProviderResponse:
    if isinstance(response, AdvisoryFitProviderResponse):
        output = response.output
        if not isinstance(output, AdvisoryFitProviderOutput):
            output = AdvisoryFitProviderOutput.model_validate(output)
        return AdvisoryFitProviderResponse(
            output=output,
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
    if not isinstance(response, AdvisoryFitProviderOutput):
        response = AdvisoryFitProviderOutput.model_validate(response)
    return AdvisoryFitProviderResponse(
        output=response,
        provider="deterministic-fake",
        model="deterministic-fake",
    )


async def generate_advisory_fit(
    input_data: AdvisoryFitInput,
    *,
    provider: Provider | None = None,
) -> AdvisoryFitResult:
    """Run one bounded evidence-backed Advisory Fit analysis.

    The optional provider seam is used by deterministic contract tests. The
    HTTP endpoint uses the provider selected by ``AI_PROVIDER`` by default.
    """
    if not input_data.allowed_evidence_ids:
        return AdvisoryFitResult(status="NOT_STARTED_NO_ASSESSABLE_EVIDENCE")

    started_at = time.perf_counter()
    try:
        response = _coerce_provider_response(
            await _invoke_with_retries(provider or _invoke_provider, input_data)
        )
    except AdvisoryFitProviderSystemLimit:
        return AdvisoryFitResult(status="NOT_STARTED_SYSTEM_LIMIT")
    except AdvisoryFitProviderTimeout:
        raise
    except AdvisoryFitProviderError:
        raise
    except asyncio.TimeoutError as exc:
        raise AdvisoryFitProviderTimeout(
            "Advisory Fit provider timed out"
        ) from exc
    except ValidationError as exc:
        raise AdvisoryFitProviderError(
            "Advisory Fit provider returned invalid output"
        ) from exc
    except Exception as exc:
        raise AdvisoryFitProviderError("Advisory Fit provider failed") from exc

    try:
        _validate_provider_coverage(response.output, input_data)
        latency_ms = max(0, round((time.perf_counter() - started_at) * 1000))
        metadata = AdvisoryFitMetadata(
            provider=response.provider,
            model=response.model,
            promptVersion=PROMPT_VERSION,
            schemaVersion=SCHEMA_VERSION,
            serviceVersion=settings.service_version,
            latencyMs=latency_ms,
            inputTokens=response.input_tokens,
            outputTokens=response.output_tokens,
        )
    except AdvisoryFitProviderError:
        raise
    except ValidationError as exc:
        raise AdvisoryFitProviderError(
            "Advisory Fit provider returned invalid metadata"
        ) from exc
    return AdvisoryFitResult(
        status="COMPLETED",
        findings=response.output.findings,
        metadata=metadata,
    )
