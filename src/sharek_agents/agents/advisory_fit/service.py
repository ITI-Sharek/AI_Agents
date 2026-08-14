from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable

import httpx
from groq import APIStatusError, RateLimitError
from langchain_groq import ChatGroq
from pydantic import ValidationError

from .prompts import SYSTEM_PROMPT, render_advisory_fit_prompt
from .schemas import (
    AdvisoryFitInput,
    AdvisoryFitMetadata,
    AdvisoryFitProviderOutput,
    AdvisoryFitResult,
)


class AdvisoryFitProviderError(Exception):
    """Provider output cannot be accepted safely."""


class AdvisoryFitProviderTimeout(AdvisoryFitProviderError):
    """The bounded provider call timed out."""


class AdvisoryFitProviderSystemLimit(AdvisoryFitProviderError):
    """The configured provider cannot start work because of a system limit."""


Provider = Callable[[AdvisoryFitInput], Awaitable[AdvisoryFitProviderOutput]]


def _bounded_integer(
    name: str, default: int, *, minimum: int, maximum: int
) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    if value < minimum:
        return default
    return min(value, maximum)


def _bounded_timeout_seconds() -> int:
    return _bounded_integer(
        "AI_ADVISORY_FIT_TIMEOUT_SECONDS",
        60,
        minimum=1,
        maximum=180,
    )


def _is_quota_or_billing_status(error: APIStatusError) -> bool:
    if error.status_code in {402, 429}:
        return True
    body = error.body if isinstance(error.body, dict) else {}
    detail = body.get("error", body)
    if not isinstance(detail, dict):
        return False
    values = " ".join(
        str(detail.get(key, "")) for key in ("code", "type", "message")
    ).casefold()
    return error.status_code == 403 and any(
        marker in values for marker in ("quota", "billing", "credit", "payment")
    )


async def _default_provider(input_data: AdvisoryFitInput) -> AdvisoryFitProviderOutput:
    model = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
    timeout = _bounded_timeout_seconds()
    try:
        structured = ChatGroq(
            model=model,
            temperature=0,
            timeout=timeout,
            max_retries=0,
        ).with_structured_output(AdvisoryFitProviderOutput)
        result = await asyncio.wait_for(
            structured.ainvoke(
                [
                    ("system", SYSTEM_PROMPT),
                    ("human", render_advisory_fit_prompt(input_data)),
                ]
            ),
            timeout=timeout,
        )
        return AdvisoryFitProviderOutput.model_validate(result)
    except RateLimitError as exc:
        raise AdvisoryFitProviderSystemLimit("provider system limit") from exc
    except APIStatusError as exc:
        if _is_quota_or_billing_status(exc):
            raise AdvisoryFitProviderSystemLimit("provider system limit") from exc
        raise AdvisoryFitProviderError("Advisory Fit provider failed") from exc
    except asyncio.TimeoutError as exc:
        raise AdvisoryFitProviderTimeout("Advisory Fit provider timed out") from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in {402, 429}:
            raise AdvisoryFitProviderSystemLimit("provider system limit") from exc
        raise AdvisoryFitProviderError("Advisory Fit provider failed") from exc
    except ValidationError as exc:
        raise AdvisoryFitProviderError("provider returned invalid output") from exc
    except AdvisoryFitProviderError:
        raise
    except Exception as exc:
        raise AdvisoryFitProviderError("Advisory Fit provider failed") from exc


def _validate_coverage(
    output: AdvisoryFitProviderOutput, input_data: AdvisoryFitInput
) -> None:
    requirements = {item.id: item.kind for item in input_data.requirements}
    finding_ids = [item.requirement_id for item in output.findings]
    if len(finding_ids) != len(set(finding_ids)) or set(finding_ids) != set(requirements):
        raise AdvisoryFitProviderError("findings must cover each Requirement once")
    allowed = set(input_data.allowed_evidence_ids)
    for finding in output.findings:
        if finding.requirement_kind != requirements[finding.requirement_id]:
            raise AdvisoryFitProviderError("Requirement classification changed")
        if any(citation not in allowed for citation in finding.citations):
            raise AdvisoryFitProviderError("citation outside authorized evidence scope")


async def generate_advisory_fit(
    input_data: AdvisoryFitInput, *, provider: Provider | None = None
) -> AdvisoryFitResult:
    if not input_data.allowed_evidence_ids:
        return AdvisoryFitResult(status="NOT_STARTED_NO_ASSESSABLE_EVIDENCE")

    started = time.perf_counter()
    retries = _bounded_integer(
        "AI_ADVISORY_FIT_MAX_RETRIES", 1, minimum=0, maximum=1
    )
    output: AdvisoryFitProviderOutput | None = None
    for attempt in range(retries + 1):
        try:
            output = await (provider or _default_provider)(input_data)
            break
        except AdvisoryFitProviderSystemLimit:
            return AdvisoryFitResult(status="NOT_STARTED_SYSTEM_LIMIT")
        except (AdvisoryFitProviderTimeout, AdvisoryFitProviderError):
            if attempt >= retries:
                raise

    if output is None:
        raise AdvisoryFitProviderError("provider returned no output")
    try:
        output = AdvisoryFitProviderOutput.model_validate(output)
    except ValidationError as exc:
        raise AdvisoryFitProviderError("provider returned invalid output") from exc
    _validate_coverage(output, input_data)
    return AdvisoryFitResult(
        status="COMPLETED",
        findings=output.findings,
        metadata=AdvisoryFitMetadata(
            provider="groq" if provider is None else "deterministic-fake",
            model=os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
            if provider is None
            else "deterministic-fake",
            promptVersion="advisory-fit-v1",
            schemaVersion="advisory-fit-v1",
            serviceVersion=os.environ.get("AI_SERVICE_VERSION", "0.1.0"),
            latencyMs=max(0, round((time.perf_counter() - started) * 1000)),
        ),
    )
