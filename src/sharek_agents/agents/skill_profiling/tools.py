import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from urllib.parse import urlparse

from sharek_agents.agents.skill_profiling.detection import (
    detect_frameworks as _detect_frameworks,
)
from sharek_agents.common.logging import get_logger
from sharek_agents.shared_tools.github_client import GithubClient


_EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
}

_client: GithubClient | None = None


async def _get_client() -> GithubClient:
    global _client
    if _client is None:
        _client = GithubClient()
    return _client


_ANALYSIS_TIMEOUT = 30


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


def filter_source_files(all_files: set[str]) -> list[str]:
    return [f for f in all_files if os.path.splitext(f)[1].lower() in _EXT_LANG]


@dataclass
class _ToolResult:
    stdout: str
    stderr: str
    returncode: int


async def _run_tool(command: str, args: list[str], timeout: int | None = None) -> _ToolResult | None:
    timeout = timeout or _ANALYSIS_TIMEOUT
    try:
        proc = await asyncio.create_subprocess_exec(
            command,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        if proc:
            proc.kill()
            await proc.wait()
        return None
    except FileNotFoundError:
        return _ToolResult("", "tool_not_found", -1)
    return _ToolResult(
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
        proc.returncode,
    )


async def _run_python_analysis(abspaths: list[str], relpaths: list[str]) -> dict:
    cc_result = await _run_tool("radon", ["cc"] + abspaths + ["-s", "-j"])
    if cc_result is None:
        return {"error": "timeout", "supported": True}
    if cc_result.returncode == -1:
        return {"error": "tool_not_found", "supported": True}

    mi_result = await _run_tool("radon", ["mi"] + abspaths + ["-s", "-j"])
    if mi_result is None:
        return {"error": "timeout", "supported": True}
    if mi_result.returncode == -1:
        return {"error": "tool_not_found", "supported": True}

    pylint_result = await _run_tool("pylint", abspaths + ["--output-format=json"])
    if pylint_result is None:
        return {"error": "timeout", "supported": True}
    if pylint_result.returncode == -1:
        return {"error": "tool_not_found", "supported": True}

    complexities: list[float] = []
    if cc_result.stdout:
        try:
            cc_data = json.loads(cc_result.stdout)
            for file_data in cc_data.values():
                for block in file_data:
                    complexities.append(block["complexity"])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    mi_values: list[float] = []
    if mi_result.stdout:
        try:
            mi_data = json.loads(mi_result.stdout)
            for file_data in mi_data.values():
                for block in file_data:
                    mi_values.append(block["mi"])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    issues: list[str] = []
    pylint_score = 0.0
    if pylint_result.stdout:
        try:
            messages = json.loads(pylint_result.stdout)
            for m in messages:
                issues.append(
                    f'{m.get("path", "")}:{m.get("line", 0)} {m.get("message", "")}'
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    if pylint_result.stderr:
        m = re.search(r"rated at (\d+\.\d+)/10", pylint_result.stderr)
        if m:
            pylint_score = float(m.group(1))

    return {
        "avg_complexity": round(sum(complexities) / len(complexities), 2)
        if complexities
        else 0.0,
        "maintainability_index": round(sum(mi_values) / len(mi_values), 2)
        if mi_values
        else 0.0,
        "pylint_score": pylint_score,
        "top_issues": issues[:10],
        "files_analyzed": relpaths,
    }


async def _run_js_analysis(abspaths: list[str], relpaths: list[str]) -> dict:
    result = await _run_tool("eslint", abspaths + ["--format", "json"])
    if result is None:
        return {"error": "timeout", "supported": True}
    if result.returncode == -1:
        return {"error": "tool_not_found", "supported": True}

    error_count = 0
    warning_count = 0
    issues: list[str] = []

    if result.stdout:
        try:
            reports = json.loads(result.stdout)
            for report in reports:
                error_count += report.get("errorCount", 0)
                warning_count += report.get("warningCount", 0)
                for m in report.get("messages", []):
                    issues.append(
                        f'{report.get("filePath", "")}:{m.get("line", 0)} {m.get("message", "")}'
                    )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    return {
        "error_count": error_count,
        "warning_count": warning_count,
        "top_issues": issues[:10],
        "files_analyzed": relpaths,
    }


async def run_static_analysis(
    local_repo_path: str,
    file_paths: list[str],
    language: str,
) -> dict:
    if not file_paths:
        return {
            "supported": True,
            "skipped": True,
            "reason": "no_source_files",
        }

    abspaths = [os.path.join(local_repo_path, fp) for fp in file_paths]

    if language == "python":
        return await _run_python_analysis(abspaths, file_paths)
    if language in ("javascript", "typescript"):
        return await _run_js_analysis(abspaths, file_paths)
    return {"supported": False}


async def run_graphify(repo_url: str, max_size_mb: int = 250) -> dict:
    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp()

        result = await _run_tool("git", ["clone", "--depth=1", repo_url, tmp_dir], timeout=60)
        if result is None:
            return {"skipped": True, "reason": "clone_failed"}
        if result.returncode == -1:
            return {"error": "tool_not_found", "supported": True}
        if result.returncode != 0:
            return {"skipped": True, "reason": "clone_failed"}

        total_bytes = 0
        for dirpath, _, filenames in os.walk(tmp_dir):
            for f in filenames:
                try:
                    total_bytes += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass

        if total_bytes > max_size_mb * 1024 * 1024:
            return {"skipped": True, "reason": "repo_too_large"}

        graphify_result = await _run_tool(sys.executable, ["-m", "graphify", "update", tmp_dir], timeout=60)
        if graphify_result is None:
            return {"skipped": True, "reason": "graphify_timeout"}
        if graphify_result.returncode == -1:
            return {"error": "tool_not_found", "supported": True}

        graph_path = os.path.join(tmp_dir, "graphify-out", "graph.json")
        if not os.path.isfile(graph_path):
            return {"skipped": True, "reason": "graph_not_found"}

        with open(graph_path, "r") as f:
            return json.load(f)

    except Exception as e:
        return {"skipped": True, "reason": str(e)}
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def prune_graph_for_llm(graph_json: dict, relevant_files: list[str]) -> dict:
    if not relevant_files:
        return {"files_analyzed": [], "inherits": [], "calls": []}

    relevant_set = set(relevant_files)

    node_ids = {
        n["id"]
        for n in graph_json.get("nodes", [])
        if n.get("source_file", "") in relevant_set
    }

    inherits: list[dict] = []
    calls: list[dict] = []
    for link in graph_json.get("links", []):
        if link.get("source") in node_ids and link.get("target") in node_ids:
            entry = {"source": link["source"], "target": link["target"]}
            rel = link.get("relation", "")
            if rel == "inherits":
                inherits.append(entry)
            elif rel == "calls":
                calls.append(entry)

    result: dict = {
        "files_analyzed": sorted(relevant_set),
        "inherits": inherits,
        "calls": calls,
    }

    while len(json.dumps(result)) / 4 > 2000 and (inherits or calls):
        degree: dict[str, int] = {}
        for e in inherits + calls:
            degree[e["source"]] = degree.get(e["source"], 0) + 1
            degree[e["target"]] = degree.get(e["target"], 0) + 1

        def edge_key(e):
            return degree.get(e["source"], 0) + degree.get(e["target"], 0)

        pool = calls if calls else inherits
        pool.sort(key=edge_key)
        pool.pop(0)
        result["inherits"] = inherits
        result["calls"] = calls

    return result


def _detect_language(file_paths: list[str]) -> str | None:
    for fp in file_paths:
        lang = _EXT_LANG.get(os.path.splitext(fp)[1].lower())
        if lang:
            return lang
    return None


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
    logger = get_logger(f"{__name__}._process_single_repo")
    try:
        meta = await get_repo_metadata(owner, repo_name)
        if meta is None:
            unresolved_repos.append({"url": original_url, "reason": "repo_not_found_or_private"})
            return None

        default_branch = meta.get("default_branch", "main")
        clone_url = meta.get("clone_url", f"https://github.com/{owner}/{repo_name}.git")

        try:
            current_tree = await get_current_file_tree(owner, repo_name, default_branch)
        except Exception:
            current_tree = set()

        async def _deps():
            try:
                dep_files = await get_dependency_files(owner, repo_name)
                return detect_frameworks(dep_files)
            except Exception:
                return {}

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

        frameworks, commit_derived_files = await asyncio.gather(
            _deps(), _commits()
        )

        relevant_files = [f for f in commit_derived_files if f in current_tree]

        async def _static():
            try:
                if not relevant_files:
                    return {
                        "supported": True,
                        "skipped": True,
                        "reason": "no_source_files",
                    }

                lang = _detect_language(relevant_files)
                if lang is None:
                    return {
                        "supported": True,
                        "skipped": True,
                        "reason": "unknown_language",
                    }

                tmp_clone = tempfile.mkdtemp()
                try:
                    clone_result = await _run_tool(
                        "git",
                        ["clone", "--depth=1", clone_url, tmp_clone],
                        timeout=60,
                    )
                    if clone_result is None:
                        return {
                            "supported": True,
                            "skipped": True,
                            "reason": "clone_failed",
                        }
                    if clone_result.returncode == -1:
                        return {"error": "tool_not_found", "supported": True}
                    if clone_result.returncode != 0:
                        return {
                            "supported": True,
                            "skipped": True,
                            "reason": "clone_failed",
                        }

                    return await run_static_analysis(tmp_clone, relevant_files, lang)
                finally:
                    shutil.rmtree(tmp_clone, ignore_errors=True)
            except Exception:
                return {"supported": True, "skipped": True, "reason": "static_error"}

        async def _graph():
            try:
                raw = await run_graphify(clone_url)
                return prune_graph_for_llm(raw, relevant_files)
            except Exception:
                return {"files_analyzed": [], "inherits": [], "calls": []}

        static_analysis, graph_relations = await asyncio.gather(
            _static(), _graph()
        )

        return {
            "name": repo_name,
            "owner": owner,
            "description": meta.get("description") or "",
            "language": meta.get("language") or "",
            "topics": meta.get("topics", []),
            "stars": meta.get("stargazers_count", 0),
            "forks": meta.get("forks_count", 0),
            "default_branch": default_branch,
            "clone_url": clone_url,
            "frameworks": frameworks,
            "static_analysis": static_analysis,
            "graph_relations": graph_relations,
            "commit_derived_files": commit_derived_files,
            "files_evaluated": relevant_files,
            "file_count": len(relevant_files),
        }
    except Exception:
        logger.exception("unhandled error processing repo %s/%s", owner, repo_name)
        return None


def detect_frameworks(dependency_files: dict[str, str]) -> dict[str, list[str]]:
    return _detect_frameworks(dependency_files)
