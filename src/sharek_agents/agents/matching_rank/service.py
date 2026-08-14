from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable

import httpx
from groq import APIStatusError
from langchain_groq import ChatGroq
from pydantic import ValidationError

from sharek_agents.config import settings

from .prompts import SYSTEM_PROMPT, render_matching_rank_prompt
from .schemas import (
    MatchingRankInput,
    MatchingRankMetadata,
    MatchingRankProviderOutput,
    MatchingRankResult,
)


class MatchingRankProviderError(Exception):
    """Provider output cannot be accepted safely."""


class MatchingRankProviderTimeout(MatchingRankProviderError):
    """The bounded provider call timed out."""


Provider = Callable[[MatchingRankInput], Awaitable[MatchingRankProviderOutput]]


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
        "AI_MATCHING_RANK_TIMEOUT_SECONDS", 30, minimum=1, maximum=120
    )


async def _default_provider(
    input_data: MatchingRankInput,
) -> MatchingRankProviderOutput:
    model = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
    timeout = _bounded_timeout_seconds()
    try:
        # No tools bound, deliberately. The agent needs nothing beyond the
        # shortlist it was handed, so an injected "fetch this URL" has no
        # mechanism to act through even if the model were persuaded by it.
        structured = ChatGroq(
            model=model,
            temperature=0,
            timeout=timeout,
            max_retries=0,
        ).with_structured_output(MatchingRankProviderOutput)
        result = await asyncio.wait_for(
            structured.ainvoke(
                [
                    ("system", SYSTEM_PROMPT),
                    ("human", render_matching_rank_prompt(input_data)),
                ]
            ),
            timeout=timeout,
        )
        return MatchingRankProviderOutput.model_validate(result)
    except asyncio.TimeoutError as exc:
        raise MatchingRankProviderTimeout("matching rank provider timed out") from exc
    except APIStatusError as exc:
        raise MatchingRankProviderError("matching rank provider failed") from exc
    except httpx.HTTPStatusError as exc:
        raise MatchingRankProviderError("matching rank provider failed") from exc
    except ValidationError as exc:
        raise MatchingRankProviderError("provider returned invalid output") from exc
    except MatchingRankProviderError:
        raise
    except Exception as exc:
        raise MatchingRankProviderError("matching rank provider failed") from exc


def _assert_is_permutation(
    input_data: MatchingRankInput, output: MatchingRankProviderOutput
) -> None:
    """The whole safety property of this endpoint, in one function.

    A ranker may **reorder and explain**. It may not add a request the backend's
    exclusions rejected, drop one it chose, or repeat one to push another out of
    a capped list. Checking set equality *and* length catches all three: a
    duplicate makes the lengths differ even when the sets match.

    Enforced here as well as in the backend on purpose. The backend's check is
    the one that protects the contributor; this one means a provider fault is
    reported as a 502 the caller can retry, rather than as a silently-discarded
    response that looks to the backend like the model simply had nothing to add.
    """
    expected = {candidate.request_id for candidate in input_data.candidates}
    returned = [match.request_id for match in output.matches]

    if len(returned) != len(expected):
        raise MatchingRankProviderError(
            "provider returned a different number of matches than it was given"
        )
    if set(returned) != expected:
        raise MatchingRankProviderError(
            "provider returned a different set of requestIds than it was given"
        )


async def rank_matches(
    input_data: MatchingRankInput, *, provider: Provider | None = None
) -> MatchingRankResult:
    started = time.perf_counter()
    retries = _bounded_integer("AI_MATCHING_RANK_MAX_RETRIES", 1, minimum=0, maximum=1)

    output: MatchingRankProviderOutput | None = None
    for attempt in range(retries + 1):
        try:
            output = await (provider or _default_provider)(input_data)
            break
        except (MatchingRankProviderTimeout, MatchingRankProviderError):
            # Never a partial order on the way out. Half a shortlist is worse
            # than none, because the backend cannot tell it is half and would
            # show a contributor a list missing matches it had already found.
            if attempt >= retries:
                raise

    if output is None:
        raise MatchingRankProviderError("provider returned no output")
    try:
        # Revalidated even when a caller supplied the provider, so a test double
        # cannot inject a shape the real path would have refused.
        output = MatchingRankProviderOutput.model_validate(output)
    except ValidationError as exc:
        raise MatchingRankProviderError("provider returned invalid output") from exc

    _assert_is_permutation(input_data, output)

    return MatchingRankResult(
        matches=output.matches,
        metadata=MatchingRankMetadata(
            provider=os.environ.get("LLM_PROVIDER", "groq"),
            model=os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"),
            prompt_version="matching-rank-v1",
            schema_version="matching-rank-v1",
            service_version=settings.service_version,
            latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
        ),
    )
