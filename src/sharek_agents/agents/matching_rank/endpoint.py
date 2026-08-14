from fastapi import HTTPException, status

from .schemas import MatchingRankInput, MatchingRankResult
from .service import (
    MatchingRankProviderError,
    MatchingRankProviderTimeout,
    rank_matches,
)


async def analyze_matching_rank(body: MatchingRankInput) -> MatchingRankResult:
    try:
        return await rank_matches(body)
    except MatchingRankProviderTimeout as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Matching rank provider timed out",
        ) from exc
    except MatchingRankProviderError as exc:
        # Deliberately opaque. The caller learns the attempt failed and is
        # retriable; it learns nothing about the provider, the model, or what
        # the model actually returned.
        #
        # Either error is safe for the backend to receive: it falls back to its
        # own deterministic order, so a contributor still sees their matches.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Matching rank provider returned an invalid response",
        ) from exc
