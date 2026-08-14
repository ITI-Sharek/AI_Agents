from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable

import httpx
from groq import APIStatusError
from langchain_groq import ChatGroq
from pydantic import ValidationError

from .prompts import SYSTEM_PROMPT, render_requirement_inference_prompt
from .schemas import (
    MAX_INFERRED_SKILLS,
    InferredSkillRequirement,
    RequirementInferenceInput,
    RequirementInferenceMetadata,
    RequirementInferenceProviderOutput,
    RequirementInferenceResult,
)


class RequirementInferenceProviderError(Exception):
    """Provider output cannot be accepted safely."""


class RequirementInferenceProviderTimeout(RequirementInferenceProviderError):
    """The bounded provider call timed out."""


Provider = Callable[
    [RequirementInferenceInput], Awaitable[RequirementInferenceProviderOutput]
]


def _bounded_integer(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    if value < minimum:
        return default
    return min(value, maximum)


def _bounded_timeout_seconds() -> int:
    return _bounded_integer(
        "AI_REQUIREMENT_INFERENCE_TIMEOUT_SECONDS",
        45,
        minimum=1,
        maximum=180,
    )


async def _default_provider(
    input_data: RequirementInferenceInput,
) -> RequirementInferenceProviderOutput:
    model = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
    timeout = _bounded_timeout_seconds()
    try:
        # No tools are bound to this client, and none ever should be. The agent
        # needs nothing beyond the text it was handed, so an injected
        # "fetch this URL" instruction has no mechanism to act through even if
        # the model were persuaded by it.
        structured = ChatGroq(
            model=model,
            temperature=0,
            timeout=timeout,
            max_retries=0,
        ).with_structured_output(RequirementInferenceProviderOutput)
        result = await asyncio.wait_for(
            structured.ainvoke(
                [
                    ("system", SYSTEM_PROMPT),
                    ("human", render_requirement_inference_prompt(input_data)),
                ]
            ),
            timeout=timeout,
        )
        return RequirementInferenceProviderOutput.model_validate(result)
    except asyncio.TimeoutError as exc:
        raise RequirementInferenceProviderTimeout(
            "requirement inference provider timed out"
        ) from exc
    except APIStatusError as exc:
        raise RequirementInferenceProviderError(
            "requirement inference provider failed"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise RequirementInferenceProviderError(
            "requirement inference provider failed"
        ) from exc
    except ValidationError as exc:
        raise RequirementInferenceProviderError(
            "provider returned invalid output"
        ) from exc
    except RequirementInferenceProviderError:
        raise
    except Exception as exc:
        raise RequirementInferenceProviderError(
            "requirement inference provider failed"
        ) from exc


def _collapse_and_cap(
    skills: list[InferredSkillRequirement],
) -> list[InferredSkillRequirement]:
    """One row per skill, in the order the model ranked them, capped at 15.

    **Duplicates:** the first occurrence wins. A model that names one technology
    twice has made an error, and the two rows usually say the same thing in two
    spellings; keeping the first is deterministic, preserves the model's own
    priority order, and is explainable. Where the duplicates genuinely conflict
    there is no correct automatic answer — which is survivable here precisely
    because this is a draft: the owner reviews and overrides the set before the
    Request can be published, and only then does it become a bar anyone is
    measured against.

    **Cap:** truncation rather than rejection. A verbose answer is still a
    useful starting point for the owner, and turning it into a 502 would leave
    them with nothing to edit.

    Names are already lowercase and space-collapsed by the schema, so the key
    below compares what the caller will actually receive.
    """
    seen: set[str] = set()
    collapsed: list[InferredSkillRequirement] = []
    for skill in skills:
        if skill.skill_name in seen:
            continue
        seen.add(skill.skill_name)
        collapsed.append(skill)
        if len(collapsed) == MAX_INFERRED_SKILLS:
            break
    return collapsed


async def infer_requirements(
    input_data: RequirementInferenceInput, *, provider: Provider | None = None
) -> RequirementInferenceResult:
    started = time.perf_counter()
    retries = _bounded_integer(
        "AI_REQUIREMENT_INFERENCE_MAX_RETRIES", 1, minimum=0, maximum=1
    )

    output: RequirementInferenceProviderOutput | None = None
    for attempt in range(retries + 1):
        try:
            output = await (provider or _default_provider)(input_data)
            break
        except (
            RequirementInferenceProviderTimeout,
            RequirementInferenceProviderError,
        ):
            # No partial set on the way out: a retry either produces a complete
            # validated answer or the error propagates. Half a bar is worse than
            # no bar, because the owner cannot tell it is half.
            if attempt >= retries:
                raise

    if output is None:
        raise RequirementInferenceProviderError("provider returned no output")
    try:
        # Revalidated even when a caller supplied the provider, so a test double
        # cannot inject a shape the real path would have refused.
        output = RequirementInferenceProviderOutput.model_validate(output)
    except ValidationError as exc:
        raise RequirementInferenceProviderError(
            "provider returned invalid output"
        ) from exc

    return RequirementInferenceResult(
        skills=_collapse_and_cap(output.skills),
        metadata=RequirementInferenceMetadata(
            provider="groq" if provider is None else "deterministic-fake",
            model=os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
            if provider is None
            else "deterministic-fake",
            promptVersion="requirement-inference-v1",
            schemaVersion="requirement-inference-v1",
            serviceVersion=os.environ.get("AI_SERVICE_VERSION", "0.1.0"),
            latencyMs=max(0, round((time.perf_counter() - started) * 1000)),
        ),
    )
