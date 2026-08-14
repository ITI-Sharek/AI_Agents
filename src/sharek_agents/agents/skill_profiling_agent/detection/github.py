"""Minimal GitHub REST API access for the Detection Agent Tool.

Only the endpoints this tool needs are implemented:

* ``GET /repos/{owner}/{repo}``          -> default branch resolution
* ``GET /repos/{owner}/{repo}/git/trees/{ref}?recursive=1`` -> file discovery
* ``GET /repos/{owner}/{repo}/contents/{path}`` (raw accept) -> file content

All requests carry the caller-provided GitHub token as a bearer token.
The token is never included in exceptions, logs, or returned values.

An ``httpx.AsyncClient`` may be injected for tests (e.g. with
``httpx.MockTransport``); injected clients are not closed by the client.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

GITHUB_API_BASE_URL = "https://api.github.com"
USER_AGENT = "sharek-skill-profiling-agent/1.0"
API_VERSION_HEADER = "2022-11-28"

_CONTENTS_RAW_ACCEPT = "application/vnd.github.raw+json"
_REST_ACCEPT = "application/vnd.github+json"


class RepositoryAccessError(Exception):
    """Safe error raised when a GitHub API operation fails.

    Messages are intentionally sanitized: they describe the outcome
    without echoing request details or credentials.
    """


@dataclass(frozen=True)
class RepositoryContext:
    """Repository information available to the detection tool.

    ``default_branches`` maps ``owner/name`` to the repository's default
    branch when it is already known from request context; it is used to
    skip the repository-metadata API call.
    """

    github_token: str
    default_branches: dict[str, str] | None = None


class GitHubClient:
    """Thin GitHub REST client used by the Detection Agent Tool."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = GITHUB_API_BASE_URL,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._http: httpx.AsyncClient | None = http_client
        self._owns_http = http_client is None

    @property
    def base_url(self) -> str:
        return self._base_url

    async def close(self) -> None:
        """Release the HTTP session (injected clients are not closed)."""
        if self._http is not None and self._owns_http:
            await self._http.aclose()
        self._http = None

    async def get_default_branch(self, full_name: str) -> str:
        """Resolve the repository's default branch via the API."""
        response = await self._get(
            f"/repos/{_quote(full_name)}",
            accept=_REST_ACCEPT,
        )
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("default_branch"), str):
            raise RepositoryAccessError(
                f"GitHub API returned no default branch for '{full_name}'"
            )
        return data["default_branch"]

    async def list_dependency_file_paths(
        self,
        full_name: str,
        branch: str,
        candidates: frozenset[str] | set[str],
    ) -> list[str]:
        """Discover dependency-file paths in the repository tree.

        Uses the recursive git tree API; when the tree is truncated or
        unavailable, falls back to listing the repository root contents.
        Returns matching paths sorted deterministically.
        """
        paths: set[str] = set()
        try:
            response = await self._get(
                f"/repos/{_quote(full_name)}/git/trees/{_quote(branch)}",
                params={"recursive": "1"},
                accept=_REST_ACCEPT,
            )
            data = response.json()
            if isinstance(data, dict):
                tree = data.get("tree")
                if isinstance(tree, list):
                    for node in tree:
                        if isinstance(node, dict) and node.get("type") == "blob":
                            path = node.get("path")
                            if isinstance(path, str) and _is_candidate(path, candidates):
                                paths.add(path)
                    if data.get("truncated") is True:
                        paths.update(
                            await self._root_candidate_paths(full_name, candidates)
                        )
        except RepositoryAccessError:
            paths.update(await self._root_candidate_paths(full_name, candidates))
        return sorted(paths)

    async def _root_candidate_paths(
        self,
        full_name: str,
        candidates: frozenset[str] | set[str],
    ) -> list[str]:
        response = await self._get(
            f"/repos/{_quote(full_name)}/contents/",
            accept=_REST_ACCEPT,
        )
        data = response.json()
        if not isinstance(data, list):
            return []
        return sorted(
            entry["name"]
            for entry in data
            if isinstance(entry, dict)
            and isinstance(entry.get("name"), str)
            and _is_candidate(entry["name"], candidates)
        )

    async def get_file_content(self, full_name: str, path: str) -> str:
        """Return the raw text content of one repository file."""
        response = await self._get(
            f"/repos/{_quote(full_name)}/contents/{_quote(path)}",
            accept=_CONTENTS_RAW_ACCEPT,
        )
        content = response.text
        if not isinstance(content, str):
            raise RepositoryAccessError(
                f"GitHub API returned no content for '{path}'"
            )
        return content

    # ── Internal helpers ────────────────────────────────────────────────────

    def _http_client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._auth_headers(),
                follow_redirects=True,
                timeout=30.0,
            )
        return self._http

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Accept": _REST_ACCEPT,
            "Authorization": f"Bearer {self._token}",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": API_VERSION_HEADER,
        }

    async def _get(
        self,
        url: str,
        *,
        accept: str,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        client = self._http_client()
        headers = {**self._auth_headers(), "Accept": accept}
        try:
            response = await client.get(
                url,
                params=params,
                headers=headers,
            )
        except httpx.RequestError as exc:
            raise RepositoryAccessError(
                "GitHub API request failed"
            ) from exc
        if response.status_code == 404:
            raise RepositoryAccessError("GitHub repository or file not found")
        if response.status_code in (401, 403):
            raise RepositoryAccessError(
                "GitHub API authorization failed or rate limit exceeded"
            )
        if response.status_code == 429:
            raise RepositoryAccessError("GitHub API rate limit exceeded")
        if response.status_code >= 400:
            raise RepositoryAccessError(
                f"GitHub API returned HTTP {response.status_code}"
            )
        return response


def _quote(value: str) -> str:
    """Percent-encode each path segment of a GitHub REST URL.

    The ``/`` separators between ``owner``, ``repo`` and file-path
    segments are preserved (GitHub's canonical REST form).
    """
    from urllib.parse import quote as _url_quote

    return "/".join(_url_quote(segment, safe="") for segment in value.split("/"))


def _is_candidate(path: str, candidates: frozenset[str] | set[str]) -> bool:
    """True when a tree path is a known dependency manifest file."""
    basename = path.rsplit("/", 1)[-1]
    if basename in candidates:
        return True
    return basename.endswith(".csproj")
