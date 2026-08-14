"""What `/matching/rank` may and may not do.

The endpoint's value is not the ordering — the backend already has a usable one.
Its value is that it cannot do anything *else*: it cannot surface a request the
backend excluded, drop one it chose, invent an id, or emit a number. These tests
are that boundary.

No test makes a paid provider call; every one injects a provider double.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from pydantic import ValidationError

from sharek_agents.agents.matching_rank.schemas import (
    MatchingRankInput,
    MatchingRankProviderOutput,
    RankedMatch,
)
from sharek_agents.agents.matching_rank.service import (
    MatchingRankProviderError,
    MatchingRankProviderTimeout,
    _bounded_timeout_seconds,
    rank_matches,
)
from sharek_agents.main import app


def payload() -> dict:
    return {
        "matchingRequestId": "match-1",
        "approvedSkills": [
            {"name": "NestJS", "proficiency": "advanced"},
            {"name": "PostgreSQL", "proficiency": "intermediate"},
        ],
        "candidates": [
            {
                "requestId": "request-1",
                "title": "Harden the ingestion worker",
                "projectName": "Share-k API",
                "technologyTags": ["NestJS", "PostgreSQL"],
                "requirementTexts": ["Write tested NestJS services."],
                "matchedSkills": [{"name": "NestJS", "proficiency": "advanced"}],
                "confidence": "HIGH",
                "deterministicRank": 1,
            },
            {
                "requestId": "request-2",
                "title": "Tune the read replicas",
                "projectName": "Share-k API",
                "technologyTags": ["PostgreSQL"],
                "requirementTexts": ["Deep PostgreSQL experience."],
                "matchedSkills": [
                    {"name": "PostgreSQL", "proficiency": "intermediate"}
                ],
                "confidence": "LOW",
                "deterministicRank": 2,
            },
        ],
        "contractVersion": "matching-rank-v1",
    }


def parsed() -> MatchingRankInput:
    return MatchingRankInput.model_validate(payload())


def reordered() -> MatchingRankProviderOutput:
    return MatchingRankProviderOutput(
        matches=[
            RankedMatch(
                requestId="request-2",
                whyThisMatches="Your approved PostgreSQL is what this one leans on.",
            ),
            RankedMatch(
                requestId="request-1",
                whyThisMatches="Your approved NestJS matches what this asks for.",
            ),
        ]
    )


def provider_returning(output: MatchingRankProviderOutput):
    async def provider(_input: MatchingRankInput) -> MatchingRankProviderOutput:
        return output

    return provider


def provider_raising(error: Exception):
    async def provider(_input: MatchingRankInput) -> MatchingRankProviderOutput:
        raise error

    return provider


def post(path: str, *, token: str | None, body: dict):
    async def request():
        transport = httpx.ASGITransport(app=app)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.post(path, headers=headers, json=body)

    return asyncio.run(request())


class TestReorderingIsAllowed:
    def test_a_permutation_is_returned_in_the_providers_order(self) -> None:
        result = asyncio.run(
            rank_matches(parsed(), provider=provider_returning(reordered()))
        )

        assert [match.request_id for match in result.matches] == [
            "request-2",
            "request-1",
        ]

    def test_position_is_the_only_ranking_signal(self) -> None:
        result = asyncio.run(
            rank_matches(parsed(), provider=provider_returning(reordered()))
        )

        # No rank field to disagree with the list order, and no score field at
        # all — position *is* the rank.
        serialized = result.model_dump_json()
        assert "rank" not in RankedMatch.model_fields
        assert "score" not in serialized.lower()
        assert "%" not in serialized


class TestTheProviderCannotChangeTheSet:
    def test_an_added_request_is_rejected(self) -> None:
        # The backend's exclusions rejected everything not in the input. A
        # ranker that reintroduces one must not be able to.
        output = MatchingRankProviderOutput(
            matches=[
                *reordered().matches,
                RankedMatch(requestId="request-99", whyThisMatches="Invented."),
            ]
        )

        with pytest.raises(MatchingRankProviderError):
            asyncio.run(rank_matches(parsed(), provider=provider_returning(output)))

    def test_a_dropped_request_is_rejected(self) -> None:
        output = MatchingRankProviderOutput(matches=[reordered().matches[0]])

        with pytest.raises(MatchingRankProviderError):
            asyncio.run(rank_matches(parsed(), provider=provider_returning(output)))

    def test_a_duplicated_request_is_rejected(self) -> None:
        # Same set, same length — caught only because length is checked against
        # the expected count rather than against the returned set.
        first = reordered().matches[0]
        output = MatchingRankProviderOutput(matches=[first, first])

        with pytest.raises(MatchingRankProviderError):
            asyncio.run(rank_matches(parsed(), provider=provider_returning(output)))

    def test_a_substituted_id_is_rejected(self) -> None:
        output = MatchingRankProviderOutput(
            matches=[
                RankedMatch(requestId="request-1", whyThisMatches="Fine."),
                RankedMatch(requestId="request-3", whyThisMatches="Not in the input."),
            ]
        )

        with pytest.raises(MatchingRankProviderError):
            asyncio.run(rank_matches(parsed(), provider=provider_returning(output)))


class TestNoNumberReachesTheContributor:
    def test_a_narrative_containing_a_percentage_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            RankedMatch(requestId="request-1", whyThisMatches="A 92% match for you.")

    def test_a_narrative_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            RankedMatch(requestId="request-1", whyThisMatches="x" * 301)

    def test_the_input_schema_refuses_a_score_field(self) -> None:
        body = payload()
        body["candidates"][0]["matchScore"] = 0.91

        with pytest.raises(ValidationError):
            MatchingRankInput.model_validate(body)

    def test_the_output_schema_refuses_a_score_field(self) -> None:
        with pytest.raises(ValidationError):
            RankedMatch.model_validate(
                {
                    "requestId": "request-1",
                    "whyThisMatches": "Fine.",
                    "matchScore": 0.91,
                }
            )


class TestNoVerdictReachesTheContributor:
    def test_there_is_nowhere_to_put_an_eligibility_conclusion(self) -> None:
        # ADR 0001's split: the agent describes, the backend decides. Asserted
        # against the schema rather than the prompt, because a prompt is a
        # request and a schema is a guarantee.
        for forbidden in ("eligible", "blocked", "verdict", "recommended"):
            assert forbidden not in RankedMatch.model_fields

    def test_the_input_carries_no_contributor_identity(self) -> None:
        for forbidden in ("contributor_id", "user_id", "email"):
            assert forbidden not in MatchingRankInput.model_fields

        body = payload()
        body["contributorId"] = "11111111-1111-4111-8111-111111111111"
        with pytest.raises(ValidationError):
            MatchingRankInput.model_validate(body)


class TestProviderFailure:
    def test_a_timeout_propagates_as_a_timeout(self) -> None:
        with pytest.raises(MatchingRankProviderTimeout):
            asyncio.run(
                rank_matches(
                    parsed(),
                    provider=provider_raising(MatchingRankProviderTimeout("slow")),
                )
            )

    def test_a_failure_never_yields_a_partial_order(self) -> None:
        # The backend can fall back to its own order; it cannot detect a
        # shortlist that is quietly missing entries.
        with pytest.raises(MatchingRankProviderError):
            asyncio.run(
                rank_matches(
                    parsed(),
                    provider=provider_raising(MatchingRankProviderError("bad")),
                )
            )

    @pytest.mark.parametrize(
        ("configured", "expected"),
        [("0", 30), ("-5", 30), ("nonsense", 30), ("15", 15), ("9999", 120)],
    )
    def test_the_timeout_is_bounded(
        self, monkeypatch, configured: str, expected: int
    ) -> None:
        monkeypatch.setenv("AI_MATCHING_RANK_TIMEOUT_SECONDS", configured)
        assert _bounded_timeout_seconds() == expected


class TestTheEndpoint:
    def test_it_requires_the_shared_service_token(self, monkeypatch) -> None:
        monkeypatch.setenv("AI_SERVICE_AUTH_TOKEN", "service-secret")

        assert post("/matching/rank", token=None, body=payload()).status_code == 401
        assert post("/matching/rank", token="wrong", body=payload()).status_code == 401

    def test_it_reports_unconfigured_auth_rather_than_allowing_the_call(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("AI_SERVICE_AUTH_TOKEN", raising=False)

        assert (
            post("/matching/rank", token="anything", body=payload()).status_code == 503
        )

    def test_an_empty_shortlist_is_refused(self, monkeypatch) -> None:
        # Nothing to rank is a caller error, not an empty success: the backend
        # short-circuits before calling at all.
        monkeypatch.setenv("AI_SERVICE_AUTH_TOKEN", "service-secret")
        body = payload()
        body["candidates"] = []

        assert (
            post("/matching/rank", token="service-secret", body=body).status_code == 422
        )

    def test_a_repeated_candidate_is_refused(self, monkeypatch) -> None:
        monkeypatch.setenv("AI_SERVICE_AUTH_TOKEN", "service-secret")
        body = payload()
        body["candidates"].append(body["candidates"][0])

        assert (
            post("/matching/rank", token="service-secret", body=body).status_code == 422
        )
