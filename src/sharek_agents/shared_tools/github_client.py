import base64
from typing import Any

import httpx

from sharek_agents.config import settings


class GithubClient:
    BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None):
        self.token = token or settings.github_token
        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "sharek-ai-agents/1.0",
        }

    async def _get(self, url: str, params: dict[str, Any] | None = None) -> dict | list | None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=self._headers, params=params)
            if resp.status_code in (403, 404):
                return None
            resp.raise_for_status()
            return resp.json()

    async def get_user_repos(self, username: str, max_repos: int = 10) -> list[dict]:
        url = f"{self.BASE_URL}/users/{username}/repos"
        data = await self._get(url, {"sort": "pushed", "per_page": max_repos, "type": "public"})
        return data if isinstance(data, list) else []

    async def get_commits(self, owner: str, repo: str, *, author: str, per_page: int = 30) -> list[dict]:
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/commits"
        data = await self._get(url, {"author": author, "per_page": per_page})
        return data if isinstance(data, list) else []

    async def get_commit(self, owner: str, repo: str, sha: str) -> dict | None:
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/commits/{sha}"
        data = await self._get(url)
        return data if isinstance(data, dict) else None

    async def get_content(self, owner: str, repo: str, path: str) -> str | None:
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/contents/{path}"
        data = await self._get(url)
        if not isinstance(data, dict) or "content" not in data:
            return None
        return base64.b64decode(data["content"]).decode("utf-8")

    async def get_repo(self, owner: str, repo: str) -> dict | None:
        url = f"{self.BASE_URL}/repos/{owner}/{repo}"
        data = await self._get(url)
        return data if isinstance(data, dict) else None

    async def get_tree(self, owner: str, repo: str, branch: str, recursive: bool = False) -> dict | None:
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/git/trees/{branch}"
        params = {"recursive": "1"} if recursive else None
        data = await self._get(url, params)
        return data if isinstance(data, dict) else None
