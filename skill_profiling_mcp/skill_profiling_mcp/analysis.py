"""Static analysis and Graphify evidence, executed inside the sandbox.

Phase 6: ``analyze_static`` and ``analyze_graph`` analyze inside the
sandbox workspace. Phase 21: both analyzers REUSE the sandbox container
created by ``acquire_repository`` — no new container, no re-clone, and no
scope rebuild. Phase 22: ``analyze_static`` targets the FULL repository
workspace (``/workspace/repo``) — there is NO contributor file filtering
before static analysis; Graphify also runs against the FULL repository
inside the sandbox and only the contributor-related nodes/relations are
selected for evidence (preserving the contributor's architectural
relationships with the rest of the repository).

Phase 24 (orchestration): the ``analyze_contributor_repository`` MCP
tool owns the complete workflow. ``StaticAnalyzer.analyze_scope`` runs
static analysis on the contributor-owned files ONLY (the registered
scope ``/workspace/scope`` — never the full repository).
``GraphAnalyzer.analyze_graph_extract`` runs Graphify on the FULL
repository (extraction only, no scope consulted) and
``GraphAnalyzer.analyze_graph_select`` filters the resulting graph to
the contributor scope; the orchestration guarantees that graph filtering
always happens after Graphify and after the scope manifest exists.

Security model (mirrors Phases 4-5):

  * Docker is mandatory; there is NO host-side execution fallback.
  * No host-side git, static-analysis tools, or Graphify.
  * The host executes only fixed ``docker`` CLI commands (exec array, no
    shell).
  * The registered sandbox container is validated before every call
    (fail closed when Docker is unreachable or the sandbox is gone).
  * ``sandbox_identifier`` and ``workspace_path`` must belong to a
    registered workspace or the call fails closed: static analysis
    requires the full repository workspace, Graphify requires the
    registered contributor scope.
  * Resource limits, bounded timeouts, bounded evidence output, and
    cleanup: the sandbox container is removed when filtering/analysis
    fails; it is kept running on success so the pipeline can reuse it.
  * Unsupported languages return a deterministic structured
    ``unsupported_language`` evidence without ever touching Docker.

The actual tool execution (radon, pylint, eslint, checkstyle, gocyclo,
golangci-lint, clippy-driver, rubocop, cppcheck, clang-tidy, phpcs,
detekt, swiftlint, dart, sqlfluff, dotnet, semgrep, graphify)
happens INSIDE the container via ``skill_profiling_mcp.analysis_runner``;
the host only parses the bounded JSON evidence emitted on stdout.
Semgrep is an ADDITIONAL local analyzer for python/javascript/typescript:
it runs only inside the image, offline, with a bundled ruleset and no
API/cloud/API key, and its findings are merged into the same bounded
findings list — it never replaces radon, pylint, or eslint. TypeScript
uses the same ESLint adapter as JavaScript but with the in-image
``@typescript-eslint`` parser/plugin stack so ``.ts``/``.tsx`` files are
genuinely parsed as TypeScript. Vue single-file components reuse the
same ESLint installation with ``eslint-plugin-vue`` and a dedicated
in-image config (``vue-eslint-parser`` routes ``<script lang="ts">``
blocks to the in-image TypeScript stack). No language detection is
performed: the language supplied in the request is used directly.
"""

import json
import logging
from collections.abc import Callable

from skill_profiling_mcp.registry import SandboxLease, SandboxRegistry, sandbox_registry
from skill_profiling_mcp.sandbox import (
    ANALYSIS_IMAGE,
    DOCKER_BIN,
    DOCKER_INFO_TIMEOUT_SECONDS,
    CommandResult,
    SandboxError,
    _CommandTimeout,
    _run_command,
    _sanitize_error,
    ensure_sandbox_running,
)
from skill_profiling_mcp.security import REPO_WORKSPACE_PATH, SCOPE_WORKSPACE_PATH

logger = logging.getLogger(__name__)

ANALYSIS_TIMEOUT_SECONDS = 300
REMOVE_TIMEOUT_SECONDS = 30
MAX_ANALYSIS_OUTPUT_BYTES = 4 * 1024 * 1024

# Analyzer statuses produced by ``skill_profiling_mcp.analysis_runner``
# (via the ``parse_evidence`` payloads) that count as a successful analyzer
# run. Any real failure status (``error`` / ``timeout`` /
# ``tool_unavailable``) and any unknown or missing status fails closed.
_SUCCESSFUL_ANALYZER_STATUSES = frozenset(
    {"success", "extracted", "unsupported_language", "no_analyzable_content"}
)

# Language aliases mapped to canonical names. Only languages already
# supported by the project's static-analysis/Graphify tooling are
# registered — no new analyzers are invented.
LANGUAGE_ALIASES: dict[str, str] = {
    "python": "python",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "vue": "vue",
    "ts": "typescript",
    "java": "java",
    "go": "go",
    "golang": "go",
    "rust": "rust",
    "ruby": "ruby",
    "php": "php",
    "c": "c",
    "cpp": "cpp",
    "kotlin": "kotlin",
    "swift": "swift",
    "dart": "dart",
    "sql": "sql",
    "csharp": "csharp",
    "c#": "csharp",
}

ANALYZER_NAMES: dict[str, str] = {
    "python": "radon+pylint+semgrep",
    "javascript": "eslint+semgrep",
    "typescript": "eslint+semgrep",
    "vue": "eslint",
    "java": "checkstyle",
    "go": "gocyclo+golangci-lint",
    "rust": "clippy-driver",
    "ruby": "rubocop",
    "php": "phpcs",
    "c": "cppcheck",
    "cpp": "clang-tidy+cppcheck",
    "kotlin": "detekt",
    "swift": "swiftlint",
    "dart": "dart",
    "sql": "sqlfluff",
    "csharp": "dotnet",
}


def resolve_language(language: str) -> str | None:
    """Return the canonical language name for a supported alias.

    Returns None for unsupported languages (deterministic
    ``unsupported_language`` evidence is produced by the caller).
    """
    return LANGUAGE_ALIASES.get(language.strip().lower())


def parse_evidence(stdout: str) -> dict[str, object]:
    """Parse the bounded JSON evidence emitted by the in-container runner."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) < 2 or lines[0] != "EVIDENCE_OK":
        raise SandboxError("analysis evidence could not be parsed")
    try:
        evidence = json.loads(lines[1])
    except (json.JSONDecodeError, TypeError) as exc:
        raise SandboxError("analysis evidence could not be parsed") from exc
    if not isinstance(evidence, dict):
        raise SandboxError("analysis evidence could not be parsed")
    return evidence


class ContributorScopeAnalyzer:
    """Base: sandbox validation shared by the static and graph analyzers.

    Both analyzers execute inside the sandbox container created by
    ``acquire_repository`` (Phase 21) — no container is created here.
    """

    def __init__(
        self,
        *,
        runner: Callable[..., CommandResult] = _run_command,
        registry: SandboxRegistry | None = None,
    ) -> None:
        self._runner = runner
        self._registry = registry or sandbox_registry

    # ------------------------------------------------------------------
    # Workspace validation (fail closed)
    # ------------------------------------------------------------------

    def _repository_workspace_lease(
        self, sandbox_identifier: str, workspace_path: str
    ) -> SandboxLease:
        """Return the lease only for a registered sandbox's repository workspace.

        Used by ``analyze_static``: the only analyzable workspace is the
        FULL repository workspace (``/workspace/repo``) of a registered
        sandbox — there is no contributor filtering before static
        analysis. Fails closed on: unknown sandbox or a workspace path
        that is not the full repository workspace.
        """
        lease = self._registry.get(sandbox_identifier)
        if lease is None:
            raise SandboxError("sandbox unavailable: unknown sandbox_identifier")
        if workspace_path != REPO_WORKSPACE_PATH:
            raise SandboxError(
                "sandbox unavailable: workspace_path is not the full repository "
                f"workspace ({REPO_WORKSPACE_PATH})"
            )
        return lease

    def _lease_for(self, sandbox_identifier: str, workspace_path: str) -> SandboxLease:
        """Return the lease only when it belongs to a registered scope.

        Used by ``analyze_graph``: Graphify requires the registered
        contributor scope. Fails closed on: unknown sandbox, sandbox
        without a registered contributor scope, or a workspace path that
        does not belong to the registered scope.
        """
        lease = self._registry.get(sandbox_identifier)
        if lease is None:
            raise SandboxError("sandbox unavailable: unknown sandbox_identifier")
        if lease.scope is None:
            raise SandboxError(
                "sandbox unavailable: contributor scope is not registered"
            )
        if lease.scope.workspace_path != workspace_path:
            raise SandboxError(
                "sandbox unavailable: workspace_path does not belong to the contributor scope"
            )
        return lease

    def _ensure_sandbox(self, container: str) -> None:
        ensure_sandbox_running(
            self._runner,
            container,
            timeout=DOCKER_INFO_TIMEOUT_SECONDS,
            context="run analysis outside a sandbox",
        )

    def _remove_container(self, container: str) -> None:
        try:
            self._runner([DOCKER_BIN, "rm", "-f", container], timeout=REMOVE_TIMEOUT_SECONDS)
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug("container %s already removed", container, exc_info=True)

    # ------------------------------------------------------------------
    # Analysis execution (inside the existing sandbox container)
    # ------------------------------------------------------------------

    def _exec_evidence(self, container: str, exec_args: list[str]) -> dict[str, object]:
        """Execute the in-container analysis runner and parse its evidence.

        The command runs against the EXISTING sandbox container — the
        full repository clone lives in ``/workspace/repo`` and the
        contributor scope (``/workspace/scope``) built by
        ``filter_contributor_code`` (used by Graphify's selection).
        """
        exec_cmd = [DOCKER_BIN, "exec", container, *exec_args]
        try:
            result = self._runner(exec_cmd, timeout=ANALYSIS_TIMEOUT_SECONDS)
        except _CommandTimeout:
            raise SandboxError(
                f"analysis timed out after {ANALYSIS_TIMEOUT_SECONDS}s"
            ) from None
        if result.returncode != 0:
            raise SandboxError(
                f"analysis failed inside the sandbox: {_sanitize_error(result.stderr, None)}"
            )
        if len(result.stdout) > MAX_ANALYSIS_OUTPUT_BYTES:
            raise SandboxError("analysis output exceeds the size limit")
        return parse_evidence(result.stdout)


def _analyzer_succeeded(status: str) -> bool:
    """Derive an analyzer report's nested ``success`` from its ``status``.

    ``success`` / ``extracted`` / ``unsupported_language`` /
    ``no_analyzable_content`` are successful; ``error`` / ``timeout`` /
    ``tool_unavailable`` and any unknown or missing status fail closed
    with ``success=False``. The ``status`` field itself is never
    rewritten — it stays exactly as produced by the analyzer.
    """
    return status in _SUCCESSFUL_ANALYZER_STATUSES


def _unsupported_language_result(
    sandbox_identifier: str, workspace_path: str, language: str
) -> dict[str, object]:
    """Deterministic structured evidence for an unsupported language.

    Produced before any Docker interaction: no container, no image, no
    analysis is ever attempted for an unsupported language.
    """
    return {
        "success": True,
        "sandbox_identifier": sandbox_identifier,
        "workspace_path": workspace_path,
        "language": language,
        "status": "unsupported_language",
        "analyzer": None,
        "files_analyzed": 0,
        "complexity": None,
        "maintainability_index": None,
        "findings": [],
        "finding_count": 0,
        "finding_truncated": False,
        "severity_counts": {},
        "error_message": None,
        "message": f"unsupported language: {language}",
    }


class StaticAnalyzer(ContributorScopeAnalyzer):
    """Runs static analysis on the full repository workspace inside Docker."""

    def analyze_static(
        self, sandbox_identifier: str, workspace_path: str, language: str
    ) -> dict[str, object]:
        """Analyze the FULL repository workspace (``/workspace/repo``).

        There is NO contributor file filtering before static analysis.
        The workspace must be the full repository workspace of a
        registered sandbox. Returns bounded structured evidence. Raises
        ``DockerUnavailableError`` / ``SandboxError`` (fail closed) on
        any sandbox, workspace, or execution problem.
        """
        self._repository_workspace_lease(sandbox_identifier, workspace_path)
        canonical = resolve_language(language)
        if canonical is None:
            return _unsupported_language_result(
                sandbox_identifier, workspace_path, language
            )
        self._ensure_sandbox(sandbox_identifier)
        try:
            evidence = self._exec_evidence(
                sandbox_identifier,
                [
                    "python",
                    "-m",
                    "skill_profiling_mcp.analysis_runner",
                    "static",
                    REPO_WORKSPACE_PATH,
                    "--language",
                    canonical,
                    "--timeout",
                    str(ANALYSIS_TIMEOUT_SECONDS),
                ],
            )
            return self._static_report(
                sandbox_identifier, workspace_path, language, canonical, evidence
            )
        except Exception:
            self._remove_container(sandbox_identifier)
            raise

    def _static_report(
        self,
        sandbox_identifier: str,
        workspace_path: str,
        language: str,
        canonical: str,
        evidence: dict[str, object],
    ) -> dict[str, object]:
        status = evidence.get("status", "error")
        return {
            "success": _analyzer_succeeded(status),
            "sandbox_identifier": sandbox_identifier,
            "workspace_path": workspace_path,
            "language": evidence.get("language") or language,
            "status": status,
            "analyzer": evidence.get("analyzer") or ANALYZER_NAMES.get(canonical),
            "files_analyzed": int(evidence.get("files_analyzed", 0)),
            "complexity": evidence.get("complexity"),
            "maintainability_index": evidence.get("maintainability_index"),
            "findings": list(evidence.get("findings", [])),
            "finding_count": int(evidence.get("finding_count", 0)),
            "finding_truncated": bool(evidence.get("finding_truncated", False)),
            "severity_counts": dict(evidence.get("severity_counts", {})),
            "error_message": evidence.get("error_message"),
            "message": f"static analysis completed: {status}",
        }

    def analyze_scope(
        self, sandbox_identifier: str, language: str
    ) -> dict[str, object]:
        """Run static analysis on the contributor-owned files only.

        The analyzable workspace is the registered contributor scope
        (``/workspace/scope``) built by ``filter_contributor_code`` —
        static analysis NEVER runs on the full repository in this mode.
        Fails closed on an unknown sandbox or an unregistered
        contributor scope. Returns the same bounded static evidence
        schema as ``analyze_static`` (unsupported languages produce the
        deterministic ``unsupported_language`` result without touching
        Docker). Used by the ``analyze_contributor_repository``
        orchestration (Branch A, step 2).
        """
        self._lease_for(sandbox_identifier, SCOPE_WORKSPACE_PATH)
        canonical = resolve_language(language)
        if canonical is None:
            return _unsupported_language_result(
                sandbox_identifier, SCOPE_WORKSPACE_PATH, language
            )
        self._ensure_sandbox(sandbox_identifier)
        try:
            evidence = self._exec_evidence(
                sandbox_identifier,
                [
                    "python",
                    "-m",
                    "skill_profiling_mcp.analysis_runner",
                    "static",
                    SCOPE_WORKSPACE_PATH,
                    "--language",
                    canonical,
                    "--timeout",
                    str(ANALYSIS_TIMEOUT_SECONDS),
                ],
            )
            return self._static_report(
                sandbox_identifier, SCOPE_WORKSPACE_PATH, language, canonical, evidence
            )
        except Exception:
            self._remove_container(sandbox_identifier)
            raise


class GraphAnalyzer(ContributorScopeAnalyzer):
    """Runs Graphify on the full repository inside Docker.

    Graphify analyzes the FULL repository (``/workspace/repo``) in the
    sandbox; the in-container runner then selects the
    contributor-related nodes and relations (from the registered scope
    manifest at ``/workspace/scope.manifest``) for the bounded graph
    evidence. Architectural relationships with the rest of the
    repository are preserved in the selected relations.
    """

    def analyze_graph(
        self, sandbox_identifier: str, workspace_path: str
    ) -> dict[str, object]:
        """Run Graphify against the full repository, scoped for evidence.

        Returns bounded structured graph evidence — counts and relation
        summaries, never the source tree or raw graph data.
        """
        self._lease_for(sandbox_identifier, workspace_path)
        self._ensure_sandbox(sandbox_identifier)
        try:
            evidence = self._exec_evidence(
                sandbox_identifier,
                [
                    "python",
                    "-m",
                    "skill_profiling_mcp.analysis_runner",
                    "graph",
                    "/workspace/repo",
                    "--timeout",
                    str(ANALYSIS_TIMEOUT_SECONDS),
                ],
            )
            return self._graph_report(sandbox_identifier, workspace_path, evidence)
        except Exception:
            self._remove_container(sandbox_identifier)
            raise

    def _graph_report(
        self,
        sandbox_identifier: str,
        workspace_path: str,
        evidence: dict[str, object],
    ) -> dict[str, object]:
        status = evidence.get("status", "error")
        return {
            "success": _analyzer_succeeded(status),
            "sandbox_identifier": sandbox_identifier,
            "workspace_path": workspace_path,
            "status": status,
            "analyzer": "graphify",
            "graph_available": bool(evidence.get("graph_available", False)),
            "node_count": int(evidence.get("node_count", 0)),
            "edge_count": int(evidence.get("edge_count", 0)),
            "relations": dict(evidence.get("relations", {})),
            "relation_count": int(evidence.get("relation_count", 0)),
            "relation_truncated": bool(evidence.get("relation_truncated", False)),
            "inheritance_depth": evidence.get("inheritance_depth"),
            "coupling": evidence.get("coupling"),
            "circular_import_count": int(evidence.get("circular_import_count", 0)),
            "error_message": evidence.get("error_message"),
            "message": f"graph analysis completed: {status}",
        }

    def _extract_report(
        self,
        sandbox_identifier: str,
        evidence: dict[str, object],
    ) -> dict[str, object]:
        status = evidence.get("status", "error")
        return {
            "success": _analyzer_succeeded(status),
            "sandbox_identifier": sandbox_identifier,
            "workspace_path": REPO_WORKSPACE_PATH,
            "status": status,
            "analyzer": "graphify",
            "graph_available": bool(evidence.get("graph_available", False)),
            "node_count": int(evidence.get("node_count", 0)),
            "error_message": evidence.get("error_message"),
            "message": f"graph extraction completed: {status}",
        }

    def analyze_graph_extract(
        self, sandbox_identifier: str
    ) -> dict[str, object]:
        """Run Graphify against the FULL repository (extraction only).

        No contributor filtering is applied in this step: Graphify
        analyzes the complete repository workspace inside the sandbox
        and the full graph artifact is produced. The contributor graph
        filtering is applied afterwards via ``analyze_graph_select`` —
        the MCP orchestration owns that sequencing, so Graphify always
        runs on the full repository BEFORE graph filtering.
        """
        self._repository_workspace_lease(sandbox_identifier, REPO_WORKSPACE_PATH)
        self._ensure_sandbox(sandbox_identifier)
        try:
            evidence = self._exec_evidence(
                sandbox_identifier,
                [
                    "python",
                    "-m",
                    "skill_profiling_mcp.analysis_runner",
                    "graph_extract",
                    REPO_WORKSPACE_PATH,
                    "--timeout",
                    str(ANALYSIS_TIMEOUT_SECONDS),
                ],
            )
            return self._extract_report(sandbox_identifier, evidence)
        except Exception:
            self._remove_container(sandbox_identifier)
            raise

    def analyze_graph_select(
        self, sandbox_identifier: str
    ) -> dict[str, object]:
        """Filter the full-repository graph to the contributor scope.

        Requires the registered contributor scope (its manifest is
        written by ``filter_contributor_code`` inside the sandbox) and
        the full graph artifact produced by ``analyze_graph_extract``.
        Returns bounded contributor graph evidence — counts and relation
        summaries, never the source tree or raw graph data.
        """
        self._lease_for(sandbox_identifier, SCOPE_WORKSPACE_PATH)
        self._ensure_sandbox(sandbox_identifier)
        try:
            evidence = self._exec_evidence(
                sandbox_identifier,
                [
                    "python",
                    "-m",
                    "skill_profiling_mcp.analysis_runner",
                    "graph_select",
                    REPO_WORKSPACE_PATH,
                    "--timeout",
                    str(ANALYSIS_TIMEOUT_SECONDS),
                ],
            )
            return self._graph_report(
                sandbox_identifier, SCOPE_WORKSPACE_PATH, evidence
            )
        except Exception:
            self._remove_container(sandbox_identifier)
            raise


__all__ = [
    "ANALYSIS_IMAGE",
    "ANALYSIS_TIMEOUT_SECONDS",
    "ANALYZER_NAMES",
    "LANGUAGE_ALIASES",
    "MAX_ANALYSIS_OUTPUT_BYTES",
    "ContributorScopeAnalyzer",
    "GraphAnalyzer",
    "StaticAnalyzer",
    "parse_evidence",
    "resolve_language",
]
