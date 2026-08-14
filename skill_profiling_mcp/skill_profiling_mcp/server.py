"""Skill Profiling MCP Server.

Phase 3-5: MCP infrastructure, Docker-sandboxed repository acquisition,
and deterministic contributor ownership filtering. Phase 6: static
analysis and Graphify, executed both exclusively inside isolated Docker
containers. Phase 21: one persistent sandbox container per acquired
repository is reused by the filter and both analyzers — the repository
is cloned once per pipeline and Graphify analyzes the full repository,
selecting contributor-related evidence. Phase 22: static analysis
targets the FULL repository workspace (no contributor filtering before
static analysis); Graphify keeps its full-repository → contributor
graph selection behavior. Phase 24: the active Skill Profiling flow is
the single orchestration tool ``analyze_contributor_repository``, which
owns the workflow, state sequencing, asyncio concurrency, and errors:
after acquisition it runs (Branch A) contributor filter → static
analysis on contributor-owned files, and (Branch B) Graphify on the
full repository → contributor graph filtering, concurrently, and
combines the bounded evidence into one structured result. The LLM never
controls the ordering of these operations.

Phase 25: the same orchestration tool supports TWO analysis modes,
selected by the existing ``contributor_identifier`` argument (no new
API field was introduced): when a contributor identifier is supplied,
CONTRIBUTOR analysis runs (filter → static analysis on contributor-
owned files, Graphify on the full repository → contributor graph
selection); when it is omitted (``None``), PROJECT analysis runs —
static analysis and Graphify on the FULL repository only, with no
contributor filter, no contributor scope, and no graph selection, and
the full-repository evidence is returned directly.

The granular MCP tools (``acquire_repository``,
``filter_contributor_code``, ``analyze_static``, ``analyze_graph``)
have been REMOVED from MCP registration: ``analyze_contributor_repository``
is the ONLY repository-analysis tool exposed to the Agent, so the old
multi-step workflow cannot be recreated through ``tools/list``. The
internal implementation classes (``RepositoryAcquirer``,
``ContributorFilter``, ``StaticAnalyzer``, ``GraphAnalyzer``) remain and
are used by the orchestration tool only.

The server is standalone and must not depend on any existing agent,
service, or MCP server in the repository.
"""

import asyncio

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from skill_profiling_mcp.analysis import GraphAnalyzer, StaticAnalyzer
from skill_profiling_mcp.ownership import ContributorFilter
from skill_profiling_mcp.sandbox import (
    DockerUnavailableError,
    RepositoryAcquirer,
    SandboxError,
)
from skill_profiling_mcp.security import (
    REPO_WORKSPACE_PATH,
    SCOPE_WORKSPACE_PATH,
    InvalidRepositoryUrlError,
    RepositoryReference,
    parse_repository_url,
    validate_contributor_identifier,
    validate_github_pat,
    validate_language,
)

SERVER_NAME = "skill-profiling-mcp"
SERVER_VERSION = "0.1.0"
TRANSPORT = "streamable-http"

# Analyzer status values produced by StaticAnalyzer / GraphAnalyzer
# (through ``skill_profiling_mcp.analysis_runner``). These are the actual
# statuses the aggregation below is derived from.
_SUCCESS_ANALYZER_STATUSES = frozenset({"success", "extracted"})
_NOT_APPLICABLE_ANALYZER_STATUSES = frozenset({"unsupported_language", "no_analyzable_content"})
_FAILED_ANALYZER_STATUSES = frozenset({"error", "timeout", "tool_unavailable"})

mcp: FastMCP = FastMCP(
    name=SERVER_NAME,
    instructions=(
        "Skill Profiling MCP server. "
        "The only repository-analysis tool is the single orchestration "
        "tool `analyze_contributor_repository`, which acquires the "
        "repository and runs one deterministic, concurrent MCP call: "
        "with a `contributor_identifier` it runs the contributor filter "
        "+ static analysis (contributor-owned files) and Graphify + "
        "contributor graph filtering; without one (project analysis) it "
        "runs static analysis and Graphify on the full repository only. "
        "`get_server_context` is the server health/context tool."
    ),
)


@mcp.tool()
def get_server_context() -> dict[str, object]:
    """Return deterministic context proving the MCP server is alive and this tool was invoked."""
    return {
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "status": "alive",
        "tool": "get_server_context",
        "transport": TRANSPORT,
        "protocol": "MCP",
        "phase": "phase-3-infrastructure",
    }


def _get_acquirer() -> RepositoryAcquirer:
    """Build a fresh RepositoryAcquirer (injectable for tests)."""
    return RepositoryAcquirer()


def _analyzer_outcome(status: str | None) -> str:
    """Classify one analyzer's status for overall result aggregation.

    Returns ``"success"`` for a successful analyzer run,
    ``"not_applicable"`` when the analyzer was unsupported or had no
    analyzable content (not a real failure), and ``"failure"`` for any
    real failure (``error`` / ``timeout`` / ``tool_unavailable``) or
    any unrecognized status (fail closed).
    """
    if status in _SUCCESS_ANALYZER_STATUSES:
        return "success"
    if status in _NOT_APPLICABLE_ANALYZER_STATUSES:
        return "not_applicable"
    if status in _FAILED_ANALYZER_STATUSES:
        return "failure"
    return "failure"


def _combined_outcome(
    static_status: str | None,
    graph_extract_status: str | None,
    graph_select_status: str | None,
) -> tuple[bool, str]:
    """Derive the overall success flag and summary status from the analyzer statuses.

    * both analyzers succeed                       -> success=True, "completed"
    * one analyzer unsupported, other succeeds     -> success=True, "completed_with_partial_analysis"
    * one analyzer failed, other succeeded         -> success=False, "partial_failure"
    * both analyzers failed                        -> success=False, "failed"

    The graph analyzer spans the extract step (Graphify on the full
    repository) and the select step (contributor graph filtering); the
    graph analyzer failed when either step failed. A ``None`` status
    means the step is absent (project mode has no graph selection) and
    is ignored. Unsupported language is never treated as a failure.
    """
    static = _analyzer_outcome(static_status)
    graph = "success"
    for status in (graph_extract_status, graph_select_status):
        if status is None:
            continue
        outcome = _analyzer_outcome(status)
        if outcome == "failure":
            graph = "failure"
        elif outcome == "not_applicable" and graph == "success":
            graph = "not_applicable"

    if static == "failure" or graph == "failure":
        if static == "failure" and graph == "failure":
            return False, "failed"
        return False, "partial_failure"

    if static == "not_applicable" or graph == "not_applicable":
        return True, "completed_with_partial_analysis"

    return True, "completed"


@mcp.tool()
async def analyze_contributor_repository(
    repo_url: str,
    language: str,
    contributor_identifier: str | None = None,
    github_pat: str | None = None,
) -> dict[str, object]:
    """Analyze ONE contributor's code — or the whole project — in ONE call.

    The request selects the analysis mode through the existing
    ``contributor_identifier`` argument (no new field was added): when a
    contributor identifier is supplied, CONTRIBUTOR analysis runs (steps
    below); when it is omitted (``None``), PROJECT analysis runs and
    analyzes the repository as a whole.

    The server owns the complete workflow, state sequencing,
    concurrency, and error handling — the LLM does not control the
    ordering:

      CONTRIBUTOR mode (``contributor_identifier`` supplied):
      1. Acquire the repository inside an isolated Docker sandbox
         (repository acquisition semantics; the optional ``github_pat``
         is request-scoped, used for the single authenticated clone, and
         never returned, logged, or persisted).
      2. Run two branches CONCURRENTLY with asyncio:

         Branch A: contributor filter (deterministic ownership) → static
         analysis on the contributor-owned files only.

         Branch B: Graphify on the FULL repository → filter the
         resulting graph to the contributor scope.

         Both branches start as soon as the repository is acquired; the
         graph filtering step always runs after Graphify and after the
         contributor scope manifest exists — it does NOT wait for static
         analysis, which runs independently in parallel.
      3. Wait for the filter, the static analysis, and the Graph branch
         and combine their bounded evidence with the repository/
         contributor metadata into one structured result.

      PROJECT mode (``contributor_identifier`` omitted): acquire the
      repository, then run static analysis and Graphify on the FULL
      repository (``/workspace/repo``) CONCURRENTLY and return the
      full-repository evidence directly. No contributor filter, no
      contributor scope, and no graph selection ever run: these steps
      are never started in this mode.

    Unsupported languages produce deterministic ``unsupported_language``
    static evidence. All sandbox, workspace, and execution problems fail
    closed (``DockerUnavailableError`` / ``SandboxError`` become tool
    errors). The sandbox container is registered as a TTL-bounded lease
    by the acquirer before the clone and is removed — together with its
    lease — whenever the request fails. The overall ``success`` flag and
    ``summary.status`` are derived from the actual analyzer statuses:
    ``completed`` when both analyzers succeed,
    ``completed_with_partial_analysis`` when one is unsupported/not
    applicable and the other succeeds, ``partial_failure``
    (``success=false``) when one analyzer failed, and ``failed``
    (``success=false``) when both failed — successful analyzer results
    are preserved in all cases. Returns bounded structured evidence only
    — never source code, credentials, or raw graph data.
    """
    try:
        reference = parse_repository_url(repo_url)
        pat = validate_github_pat(github_pat)
        contributor = (
            None
            if contributor_identifier is None
            else validate_contributor_identifier(contributor_identifier)
        )
        language = validate_language(language)
    except InvalidRepositoryUrlError as exc:
        raise ToolError(f"invalid repository URL: {exc}") from exc
    except ValueError as exc:
        raise ToolError(str(exc)) from exc

    acquirer = _get_acquirer()
    try:
        acquired = await asyncio.to_thread(
            acquirer.acquire, reference.url, reference.identifier, pat
        )
    except DockerUnavailableError as exc:
        raise ToolError(str(exc)) from exc
    except SandboxError as exc:
        raise ToolError(str(exc)) from exc

    sandbox_identifier = acquired["sandbox_identifier"]

    if contributor is None:
        return await _analyze_project(
            acquirer,
            sandbox_identifier,
            reference,
            acquired,
            language,
        )

    filterer = _get_filter()
    static_analyzer = _get_static_analyzer()
    graph_analyzer = _get_graph_analyzer()

    async def _run_filter() -> dict[str, object]:
        """Contributor filter: deterministic ownership + scope manifest."""
        return await asyncio.to_thread(
            filterer.filter_contributor_code,
            reference.identifier,
            contributor,
            sandbox_identifier,
        )

    async def _run_static(filter_task: asyncio.Task) -> dict[str, object]:
        """Static analysis on owned files, once the scope manifest exists.

        Runs in parallel with the Graph branch after the filter has
        written the contributor scope.
        """
        await filter_task
        return await asyncio.to_thread(
            static_analyzer.analyze_scope, sandbox_identifier, language
        )

    async def _branch_b(filter_task: asyncio.Task) -> dict[str, object]:
        """Branch B: Graphify full repository → contributor graph filter.

        The full-repository Graphify extraction starts immediately;
        ``graph_select`` runs only when extraction succeeded (a
        ``success`` / ``extracted`` status per the existing
        ``_analyzer_outcome`` classification) AND after the contributor
        filter (NOT static analysis) has written the scope manifest
        (awaiting the filter-only ``filter_task`` is the orchestrator-
        owned sequencing gate; static analysis runs independently in
        parallel with ``graph_select``). When extraction failed
        (``error`` / ``timeout`` / ``tool_unavailable`` or any unknown
        status), its original result is preserved and the branch returns
        without executing ``graph_select`` — no wasted Docker work, and
        the existing aggregation below still reports the failure.
        """
        graph_extract_result = await asyncio.to_thread(
            graph_analyzer.analyze_graph_extract, sandbox_identifier
        )
        graph_extract_status = graph_extract_result.get("status")
        if _analyzer_outcome(graph_extract_status) != "success":
            graph_filter_result: dict[str, object] = {
                "success": False,
                "sandbox_identifier": sandbox_identifier,
                "workspace_path": SCOPE_WORKSPACE_PATH,
                "status": graph_extract_status,
                "analyzer": "graphify",
                "graph_available": False,
                "node_count": 0,
                "edge_count": 0,
                "relations": {},
                "relation_count": 0,
                "relation_truncated": False,
                "inheritance_depth": None,
                "coupling": None,
                "circular_import_count": 0,
                "error_message": (
                    "contributor graph selection skipped: full-repository "
                    f"graph extraction failed with status "
                    f"{graph_extract_status}"
                ),
                "message": (
                    "contributor graph selection skipped: "
                    f"{graph_extract_status}"
                ),
            }
            return {
                "graph_extract": graph_extract_result,
                "graph_filter": graph_filter_result,
            }
        await filter_task
        graph_filter_result = await asyncio.to_thread(
            graph_analyzer.analyze_graph_select, sandbox_identifier
        )
        return {"graph_extract": graph_extract_result, "graph_filter": graph_filter_result}

    filter_task = asyncio.create_task(_run_filter())
    static_task = asyncio.create_task(_run_static(filter_task))
    branch_b_task = asyncio.create_task(_branch_b(filter_task))

    try:
        filter_result, static_result, branch_b_result = await asyncio.gather(
            filter_task, static_task, branch_b_task
        )
    except (DockerUnavailableError, SandboxError) as exc:
        for task in (filter_task, static_task, branch_b_task):
            if not task.done():
                task.cancel()
        await asyncio.to_thread(acquirer.dispose, sandbox_identifier)
        raise ToolError(str(exc)) from exc

    ownership = filter_result
    graph_extract_result = branch_b_result["graph_extract"]
    graph_filter_result = branch_b_result["graph_filter"]

    overall_success, summary_status = _combined_outcome(
        static_result.get("status"),
        graph_extract_result.get("status"),
        graph_filter_result.get("status"),
    )

    return {
        "success": overall_success,
        "tool": "analyze_contributor_repository",
        "repository": {
            "repository_identifier": acquired["repository_identifier"],
            "repo_url": reference.url,
            "default_branch": acquired.get("default_branch"),
            "workspace_path": acquired.get("workspace_path"),
            "sandbox_identifier": sandbox_identifier,
        },
        "contributor_identifier": contributor,
        "language": language,
        "ownership": ownership,
        "static_analysis": static_result,
        "graph": {
            "full_repository_graphify": graph_extract_result,
            "contributor_graph": graph_filter_result,
        },
        "summary": {
            "status": summary_status,
            "owned_file_count": int(ownership.get("file_count", 0)),
            "owned_commit_count": int(ownership.get("owned_commit_count", 0)),
            "static_analysis_status": static_result.get("status"),
            "static_files_analyzed": int(static_result.get("files_analyzed", 0)),
            "graph_status": graph_filter_result.get("status"),
            "graph_node_count": int(graph_filter_result.get("node_count", 0)),
            "graph_edge_count": int(graph_filter_result.get("edge_count", 0)),
        },
        "message": "contributor repository analysis completed",
    }


async def _analyze_project(
    acquirer: RepositoryAcquirer,
    sandbox_identifier: str,
    reference: RepositoryReference,
    acquired: dict[str, object],
    language: str,
) -> dict[str, object]:
    """PROJECT analysis mode: analyze the repository as a whole.

    After acquisition, static analysis and Graphify both run on the
    FULL repository workspace (``/workspace/repo``), concurrently, and
    the full-repository evidence is returned directly. No contributor
    filter, no contributor scope, and no graph selection EVER run in
    this mode: those steps are never started. Sandbox and execution
    problems fail closed exactly like the contributor flow (cancel the
    pending tasks and dispose the sandbox lease).
    """
    static_analyzer = _get_static_analyzer()
    graph_analyzer = _get_graph_analyzer()

    async def _run_static() -> dict[str, object]:
        """Static analysis on the FULL repository workspace."""
        return await asyncio.to_thread(
            static_analyzer.analyze_static,
            sandbox_identifier,
            REPO_WORKSPACE_PATH,
            language,
        )

    async def _run_graph() -> dict[str, object]:
        """Graphify on the FULL repository (full graph, no selection)."""
        return await asyncio.to_thread(
            graph_analyzer.analyze_graph_extract, sandbox_identifier
        )

    static_task = asyncio.create_task(_run_static())
    graph_task = asyncio.create_task(_run_graph())

    try:
        static_result, graph_extract_result = await asyncio.gather(
            static_task, graph_task
        )
    except (DockerUnavailableError, SandboxError) as exc:
        for task in (static_task, graph_task):
            if not task.done():
                task.cancel()
        await asyncio.to_thread(acquirer.dispose, sandbox_identifier)
        raise ToolError(str(exc)) from exc

    overall_success, summary_status = _combined_outcome(
        static_result.get("status"),
        graph_extract_result.get("status"),
        None,
    )

    return {
        "success": overall_success,
        "tool": "analyze_contributor_repository",
        "repository": {
            "repository_identifier": acquired["repository_identifier"],
            "repo_url": reference.url,
            "default_branch": acquired.get("default_branch"),
            "workspace_path": acquired.get("workspace_path"),
            "sandbox_identifier": sandbox_identifier,
        },
        "contributor_identifier": None,
        "language": language,
        "ownership": None,
        "static_analysis": static_result,
        "graph": {
            "full_repository_graphify": graph_extract_result,
        },
        "summary": {
            "status": summary_status,
            "static_analysis_status": static_result.get("status"),
            "static_files_analyzed": int(static_result.get("files_analyzed", 0)),
            "graph_status": graph_extract_result.get("status"),
            "graph_node_count": int(graph_extract_result.get("node_count", 0)),
            "graph_edge_count": int(graph_extract_result.get("edge_count", 0)),
            "graph_available": bool(
                graph_extract_result.get("graph_available", False)
            ),
        },
        "message": "project repository analysis completed",
    }


def _get_filter() -> ContributorFilter:
    """Build a fresh ContributorFilter (injectable for tests)."""
    return ContributorFilter()


def _get_static_analyzer() -> StaticAnalyzer:
    """Build a fresh StaticAnalyzer (injectable for tests)."""
    return StaticAnalyzer()


def _get_graph_analyzer() -> GraphAnalyzer:
    """Build a fresh GraphAnalyzer (injectable for tests)."""
    return GraphAnalyzer()


def run(transport: str = TRANSPORT) -> None:
    """Run the MCP server over the configured network transport."""
    mcp.run(transport=transport)


__all__ = [
    "SERVER_NAME",
    "SERVER_VERSION",
    "TRANSPORT",
    "analyze_contributor_repository",
    "get_server_context",
    "mcp",
    "run",
]
