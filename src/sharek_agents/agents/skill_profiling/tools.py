import asyncio
import os
import re
from urllib.parse import urlparse

from sharek_agents.agents.skill_profiling.detection import (
    detect_frameworks as _detect_frameworks,
)
from sharek_agents.common.logging import get_logger
from sharek_agents.shared_tools.github_client import GithubClient


_client: GithubClient | None = None


async def _get_client() -> GithubClient:
    global _client
    if _client is None:
        _client = GithubClient()
    return _client


def parse_github_url(url: str) -> tuple[str, str] | None:
    parsed = urlparse(url)
    if "github.com" in parsed.netloc:
        path = parsed.path.strip("/")
        if path.endswith(".git"):
            path = path[:-4]
        parts = path.split("/")
        if len(parts) >= 2:
            return parts[0], parts[1]
    match = re.match(r"git@github\.com:([^/]+)/(.+)\.git", url)
    if match:
        return match.group(1), match.group(2)
    match = re.match(r"github\.com[:/]([^/]+)/(.+)", url)
    if match:
        repo = match.group(2)
        if repo.endswith(".git"):
            repo = repo[:-4]
        return match.group(1), repo
    return None


async def get_repo_metadata(owner: str, repo: str) -> dict | None:
    client = await _get_client()
    return await client.get_repo(owner, repo)


async def get_commits(owner: str, repo: str, username: str, max_commits: int = 30) -> list[dict]:
    client = await _get_client()
    data = await client.get_commits(owner, repo, author=username, per_page=max_commits)
    return [
        {
            "sha": c["sha"],
            "date": c["commit"]["committer"]["date"],
            "message": c["commit"]["message"],
        }
        for c in data
    ]


async def get_changed_files(owner: str, repo: str, commit_sha: str) -> list[str]:
    client = await _get_client()
    data = await client.get_commit(owner, repo, commit_sha)
    if data is None:
        return []
    return [f["filename"] for f in data.get("files", [])]


async def get_dependency_files(owner: str, repo: str) -> dict[str, str]:
    client = await _get_client()
    result: dict[str, str] = {}

    candidates = [
        "requirements.txt",
        "pyproject.toml",
        "Pipfile",
        "poetry.lock",
        "package.json",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "composer.json",
        "composer.lock",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "go.mod",
        "Cargo.toml",
        "Gemfile",
        "Gemfile.lock",
        "pubspec.yaml",
        "Package.swift",
        "Podfile",
        "Cartfile",
        "packages.config",
        "Directory.Packages.props",
    ]
    contents = await asyncio.gather(
        *(client.get_content(owner, repo, path) for path in candidates),
        return_exceptions=True,
    )
    for path, content in zip(candidates, contents):
        if isinstance(content, str):
            result[path] = content

    csproj_files: list[str] = []
    repo_info = await client.get_repo(owner, repo)
    if repo_info is not None:
        default_branch = repo_info.get("default_branch", "main")
        tree = await client.get_tree(owner, repo, default_branch)
        if tree is not None:
            for item in tree.get("tree", []):
                filename = item.get("path", "")
                if filename.endswith(".csproj"):
                    csproj_files.append(filename)

    if csproj_files:
        csproj_contents = await asyncio.gather(
            *(client.get_content(owner, repo, f) for f in csproj_files[:5]),
            return_exceptions=True,
        )
        for f, content in zip(csproj_files[:5], csproj_contents):
            if isinstance(content, str):
                result[f] = content

    return result


async def get_current_file_tree(owner: str, repo: str, default_branch: str) -> set[str]:
    client = await _get_client()
    tree = await client.get_tree(owner, repo, default_branch, recursive=True)
    if tree is None:
        return set()
    return {
        item["path"]
        for item in tree.get("tree", [])
        if item.get("type") == "blob"
    }


async def gather_all_evidence(repo_urls: list[str], github_username: str) -> dict:
    repos_data: list[dict] = []
    unresolved_repos: list[dict] = []

    valid: list[tuple[str, str, str]] = []
    for url in repo_urls:
        parsed = parse_github_url(url)
        if not parsed:
            unresolved_repos.append({"url": url, "reason": "invalid_url"})
        else:
            owner, repo_name = parsed
            valid.append((owner, repo_name, url))

    results = await asyncio.gather(
        *(_process_single_repo(owner, repo, github_username, url, unresolved_repos)
          for owner, repo, url in valid),
        return_exceptions=True,
    )

    for r in results:
        if isinstance(r, Exception):
            continue
        if r is not None:
            repos_data.append(r)

    return {
        "repo_count": len(repos_data),
        "repos": repos_data,
        "unresolved_repos": unresolved_repos,
    }


async def _process_single_repo(owner: str, repo_name: str, github_username: str, original_url: str, unresolved_repos: list[dict]) -> dict | None:
    # ── INVARIANT ────────────────────────────────────────────────────────
    # Framework/library/ORM detection (detect_frameworks) is a fixed,
    # GitHub-API-only step that always runs first and independently of
    # repository cloning or analysis-service-based static/graph analysis. It must
    # never be skipped, delayed, or made conditional on clone/analysis
    # success. Do not move this logic into the analysis service or
    # make it depend on a local clone in any future change.
    # ──────────────────────────────────────────────────────────────────────
    logger = get_logger(f"{__name__}._process_single_repo")
    try:
        meta = await get_repo_metadata(owner, repo_name)
        if meta is None:
            unresolved_repos.append({"url": original_url, "reason": "repo_not_found_or_private"})
            return None

        default_branch = meta.get("default_branch", "main")

        try:
            current_tree = await get_current_file_tree(owner, repo_name, default_branch)
        except Exception:
            current_tree = set()

        # STEP 1 — Framework/Library/ORM Detection
        # Runs first, before any clone-based analysis. Uses GitHub REST API
        # exclusively (get_dependency_files); no git clone, no analysis tools,
        # no subprocesses. Pure string/token matching against registry.
        async def _detect_frameworks_step():
            try:
                dep_files = await get_dependency_files(owner, repo_name)
                return detect_frameworks(dep_files), list(dep_files.keys())
            except Exception:
                return {}, []

        frameworks, dep_file_names = await _detect_frameworks_step()

        async def _commits():
            try:
                commits = await get_commits(owner, repo_name, github_username)
                commit_file_lists = await asyncio.gather(
                    *(get_changed_files(owner, repo_name, c["sha"]) for c in commits)
                )
                seen: set[str] = set()
                result: list[str] = []
                for flist in commit_file_lists:
                    for f in flist:
                        if f not in seen:
                            seen.add(f)
                            result.append(f)
                return result
            except Exception:
                return []

        commit_derived_files = await _commits()
        relevant_files = [f for f in commit_derived_files if f in current_tree]

        # Step 2 evidence is not produced by this code path — it is
        # sourced exclusively by the analysis service and delivered
        # via RepositoryEvidenceCapsule.static_analysis and
        # RepositoryEvidenceCapsule.graph_relations in the
        # /skill-profiles/generate contract endpoint. The legacy
        # /profile/repos endpoint returns empty sentinel values.
        return {
            "name": repo_name,
            "owner": owner,
            "description": meta.get("description") or "",
            "language": meta.get("language") or "",
            "topics": meta.get("topics", []),
            "stars": meta.get("stargazers_count", 0),
            "forks": meta.get("forks_count", 0),
            "default_branch": default_branch,
            "clone_url": meta.get("clone_url", ""),
            "frameworks": frameworks,
            "dependency_files_found": dep_file_names,
            "static_analysis": {
                "supported": True,
                "skipped": True,
                "reason": "analysis_service_owns_this",
            },
            "graph_relations": {"files_analyzed": [], "inherits": [], "calls": []},
            "commit_derived_files": commit_derived_files,
            "files_evaluated": relevant_files,
            "file_count": len(relevant_files),
        }
    except Exception:
        logger.exception("unhandled error processing repo %s/%s", owner, repo_name)
        return None


def detect_frameworks(dependency_files: dict[str, str]) -> dict[str, list[str]]:
    return _detect_frameworks(dependency_files)