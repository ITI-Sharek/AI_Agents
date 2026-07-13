import asyncio
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass

from sharek_agents.common.logging import get_logger
from sharek_agents.shared_tools.github_client import GithubClient


FRAMEWORK_SIGNATURES: dict[str, str] = {
    "requirements.txt": "python",
    "pyproject.toml": "python",
    "package.json": "javascript",
    "pom.xml": "java",
    "*.csproj": "csharp",
}

KNOWN_FRAMEWORKS: dict[str, list[str]] = {
    "python": ["fastapi", "django", "flask"],
    "javascript": ["react", "vue", "@angular/core", "express", "@nestjs/core", "next"],
    "java": ["spring", "spring-boot", "jakarta", "javax"],
    "csharp": ["aspnetcore", "entity-framework", "dapper", "serilog"],
}

_client: GithubClient | None = None


async def _get_client() -> GithubClient:
    global _client
    if _client is None:
        _client = GithubClient()
    return _client


async def get_active_repos(username: str, max_repos: int = 10) -> list[dict]:
    client = await _get_client()
    return await client.get_user_repos(username, max_repos)


async def get_commits_by_author(
    owner: str, repo: str, username: str, max_commits: int = 30
) -> list[dict]:
    client = await _get_client()
    commits = await client.get_commits(
        owner, repo, author=username, per_page=max_commits
    )
    return [
        {
            "sha": c["sha"],
            "date": c["commit"]["author"]["date"],
            "message": c["commit"]["message"],
        }
        for c in commits
    ]


async def get_changed_files(owner: str, repo: str, commit_sha: str) -> list[str]:
    client = await _get_client()
    commit = await client.get_commit(owner, repo, commit_sha)
    if commit is None:
        return []
    return [f["filename"] for f in commit.get("files", [])]


async def get_dependency_files(owner: str, repo: str) -> dict[str, str]:
    client = await _get_client()
    result: dict[str, str] = {}

    candidates = [
        "requirements.txt",
        "pyproject.toml",
        "package.json",
        "pom.xml",
    ]
    for path in candidates:
        content = await client.get_content(owner, repo, path)
        if content is not None:
            result[path] = content

    repo_info = await client.get_repo(owner, repo)
    if repo_info is not None:
        default_branch = repo_info.get("default_branch", "main")
        tree = await client.get_tree(owner, repo, default_branch)
        if tree is not None:
            for item in tree.get("tree", []):
                filename = item.get("path", "")
                if filename.endswith(".csproj") and "/" not in filename:
                    content = await client.get_content(owner, repo, filename)
                    if content is not None:
                        result[filename] = content

    return result


async def get_current_file_tree(
    owner: str, repo: str, default_branch: str
) -> set[str]:
    client = await _get_client()
    tree = await client.get_tree(owner, repo, default_branch, recursive=True)
    if tree is None:
        return set()
    return {
        item["path"]
        for item in tree.get("tree", [])
        if item.get("type") == "blob"
    }


def filter_to_current_files(
    changed_files: list[str], current_tree: set[str]
) -> list[str]:
    return [f for f in changed_files if f in current_tree]


_ANALYSIS_TIMEOUT = 30


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


async def _run_python_analysis(
    abspaths: list[str], relpaths: list[str]
) -> dict:
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

    pylint_result = await _run_tool(
        "pylint", abspaths + ["--output-format=json"]
    )
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


async def _run_js_analysis(
    abspaths: list[str], relpaths: list[str]
) -> dict:
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
            "reason": "no_current_files",
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

        graphify_result = await _run_tool("graphify", ["update", tmp_dir], timeout=60)
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


_EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
}


def _detect_language(file_paths: list[str]) -> str | None:
    for fp in file_paths:
        lang = _EXT_LANG.get(os.path.splitext(fp)[1].lower())
        if lang:
            return lang
    return None


async def gather_all_evidence(username: str) -> dict:
    repos = await get_active_repos(username)

    results = await asyncio.gather(
        *[_process_single_repo(r, username) for r in repos]
    )

    return {
        "username": username,
        "repo_count": len(repos),
        "repos": [r for r in results if r is not None],
    }


async def _process_single_repo(repo: dict, username: str) -> dict | None:
    repo_name = repo.get("name", "unknown")
    logger = get_logger(f"{__name__}._process_single_repo")
    try:
        owner = repo["owner"]["login"]
        default_branch = repo.get("default_branch", "main")

        try:
            commits = await get_commits_by_author(owner, repo_name, username)
        except Exception:
            commits = []

        if not commits:
            return None

        raw_changed_files: list[str] = []
        for c in commits:
            try:
                raw_changed_files.extend(
                    await get_changed_files(owner, repo_name, c["sha"])
                )
            except Exception:
                pass

        try:
            current_tree = await get_current_file_tree(owner, repo_name, default_branch)
        except Exception:
            current_tree = set()

        relevant_files = list(dict.fromkeys(
            filter_to_current_files(raw_changed_files, current_tree)
        ))

        async def _deps():
            try:
                dep_files = await get_dependency_files(owner, repo_name)
                return detect_frameworks(dep_files)
            except Exception:
                return {}

        async def _static():
            try:
                if not relevant_files:
                    return {
                        "supported": True,
                        "skipped": True,
                        "reason": "no_current_files",
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
                        ["clone", "--depth=1", repo["clone_url"], tmp_clone],
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
                raw = await run_graphify(repo["clone_url"])
                return prune_graph_for_llm(raw, relevant_files)
            except Exception:
                return {"files_analyzed": [], "inherits": [], "calls": []}

        frameworks, static_analysis, graph_relations = await asyncio.gather(
            _deps(), _static(), _graph()
        )

        return {
            "name": repo_name,
            "frameworks": frameworks,
            "static_analysis": static_analysis,
            "graph_relations": graph_relations,
            "commit_count": len(commits),
            "files_evaluated": relevant_files,
        }
    except Exception:
        logger.exception("unhandled error processing repo %s", repo_name)
        return None


def _signature_match(filename: str, pattern: str) -> bool:
    if pattern.startswith("*."):
        return filename.endswith(pattern[1:])
    return filename == pattern


def _tokenize(content: str) -> set[str]:
    tokens = re.split(r'[=\s,;"\'<>.:~!|&\n\r\t()]+', content.lower())
    return {t for t in tokens if t}


def detect_frameworks(dependency_files: dict[str, str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    content_by_lang: dict[str, str] = {}
    for filename, content in dependency_files.items():
        content_lower = content.lower()
        for pattern, lang in FRAMEWORK_SIGNATURES.items():
            if _signature_match(filename, pattern):
                content_by_lang.setdefault(lang, "")
                content_by_lang[lang] += "\n" + content_lower
                break
    for lang, combined in content_by_lang.items():
        tokens = _tokenize(combined)
        found: list[str] = []
        for framework in KNOWN_FRAMEWORKS.get(lang, []):
            if framework.lower() in tokens:
                found.append(framework)
        if found:
            result[lang] = found
    return result
