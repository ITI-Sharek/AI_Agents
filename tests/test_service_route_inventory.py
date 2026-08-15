"""The routes this service must keep serving, and who may call them.

Every route below is called by the NestJS backend from
`src/modules/ai/integrations/*.client.ts`. Renaming or dropping one breaks a
shipped feature in a *different repository*, where nothing would notice: each
side's tests mock the other, so the mismatch only surfaces as a 404 in a real
environment.

That is exactly how `/advisory-fit/assess` came to be renamed to
`/advisory-fit/analyze` on one branch while the backend still called the old
path. This file makes that a failing test instead of an outage.

Adding a route is deliberately allowed — the service may serve more than the
backend uses. Only losing one is a failure.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from sharek_agents.main import app


#: Paths the backend calls today. Do not remove an entry to make a test pass:
#: the backend client must be changed first, in the other repository.
BACKEND_CALLED_ROUTES = {
    "/advisory-fit/assess",
    "/contributor-matching/generate",
    "/matching/rank",
    "/material-analysis/analyze",
    "/requirements/infer",
    "/skill-gap-guidance/generate",
    "/skill-profiles/generate",
}

#: Internal routes may be staged here before a backend client consumes them.
#: Keep this separate so the inventory remains honest about the deployed
#: cross-repository contract.
SERVED_NOT_YET_CALLED: set[str] = set()

#: Every internal route must reject an unauthenticated call, called or not.
GUARDED_POST_ROUTES = sorted(BACKEND_CALLED_ROUTES | SERVED_NOT_YET_CALLED)


def served_paths() -> set[str]:
    return {route.path for route in app.routes if hasattr(route, "path")}


def post(path: str, *, token: str | None, body: dict | None = None):
    async def request():
        transport = httpx.ASGITransport(app=app)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.post(path, headers=headers, json=body or {})

    return asyncio.run(request())


def test_every_internal_route_is_served() -> None:
    missing = sorted(
        (BACKEND_CALLED_ROUTES | SERVED_NOT_YET_CALLED) - served_paths()
    )
    assert not missing, (
        f"The backend calls {missing}, which this service no longer serves. "
        "Renaming or removing one of these breaks a shipped feature in "
        "ITI-Sharek/Sharek — change the backend client first."
    )


def test_the_advisory_fit_path_is_not_renamed() -> None:
    # Called out on its own because this is the rename that actually happened.
    assert "/advisory-fit/assess" in served_paths()
    assert "/advisory-fit/analyze" not in served_paths()


def test_owner_contributor_matching_route_is_served() -> None:
    assert "/contributor-matching/generate" in served_paths()


@pytest.mark.parametrize("path", GUARDED_POST_ROUTES)
def test_route_rejects_a_caller_without_the_service_token(
    path: str, monkeypatch
) -> None:
    monkeypatch.setenv("AI_SERVICE_AUTH_TOKEN", "service-secret")

    assert post(path, token=None).status_code == 401
    assert post(path, token="wrong-token").status_code == 401


@pytest.mark.parametrize("path", GUARDED_POST_ROUTES)
def test_route_reports_unconfigured_auth_rather_than_allowing_the_call(
    path: str, monkeypatch
) -> None:
    # An unset token must fail closed with 503, never fall open to "no auth
    # required".
    monkeypatch.delenv("AI_SERVICE_AUTH_TOKEN", raising=False)

    assert post(path, token="anything").status_code == 503


def test_health_stays_open_for_liveness_probes() -> None:
    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.get("/health")

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
