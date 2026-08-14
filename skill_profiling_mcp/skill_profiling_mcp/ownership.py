"""Deterministic contributor ownership filtering inside a Docker sandbox.

Phase 5: identifies files attributable to one GitHub contributor using
git history, entirely inside a Docker sandbox container. The filtering
is deterministic: no LLM decides file ownership. The future ReAct Agent
decides WHEN to call this tool; this tool decides WHAT belongs to the
contributor.

Phase 21: the sandbox container is created by ``acquire_repository`` and
is REUSED here (and by the analyzers) so the repository is cloned once
per pipeline. Filtering no longer starts a fresh container or re-clones:
it runs ``git log`` and the scope build against the repository already
present in the registered sandbox.

Flow (each call reuses the registered sandbox container):

  1. Look up the sandbox lease registered by ``acquire_repository`` for
     ``sandbox_identifier`` (fail closed when unknown).
  2. Validate the sandbox container is still running (fail closed when
     the Docker daemon is unreachable or the container is gone).
  3. ``docker exec`` a ``git log --name-status`` walk over the cloned
     repository (no clone — the worktree already exists).
  4. The host parses the git output deterministically to compute
     ownership (paths only, never file contents).
  5. ``docker exec`` builds a contributor-scoped workspace
     (``/workspace/scope``) containing only owned files that exist in
     the worktree, and writes the owned-path manifest to
     ``/workspace/scope.manifest`` for the analyzers. The manifest is
     streamed back to the host as well.
  6. The manifest is attached to the sandbox lease as the registered
     contributor scope (Phase 6), so ``analyze_static`` /
     ``analyze_graph`` analyze exactly the contributor's files from the
     same sandbox.
  7. The container is removed on failure (never left half-built).

Ownership rule (deterministic):

  A commit is attributed to the contributor when the commit author email
  is ``<login>@users.noreply.github.com``, ``<login>@github.com``, or
  starts with ``<login>@``, or the author name equals the login
  (case-insensitive). A file is in the contributor scope when any
  attributed commit in the default-branch history changed it
  (add/modify/copy, or rename destination). Files changed only by other
  authors are excluded. Deleted paths are excluded from the scope and
  reported as counts only. Merge commits contribute no file changes
  (their combined diff is empty); the commits they merge stay attributed
  to their own authors.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from skill_profiling_mcp.registry import (
    ContributorScope,
    SandboxRegistry,
    sandbox_registry,
)
from skill_profiling_mcp.sandbox import (
    DOCKER_BIN,
    DOCKER_INFO_TIMEOUT_SECONDS,
    CommandResult,
    SandboxError,
    _CommandTimeout,
    _run_command,
    _sanitize_error,
    ensure_sandbox_running,
)
from skill_profiling_mcp.security import SCOPE_WORKSPACE_PATH

logger = logging.getLogger(__name__)

FILTER_TIMEOUT_SECONDS = 600
SCOPE_TIMEOUT_SECONDS = 120
REMOVE_TIMEOUT_SECONDS = 30
MAX_FILTER_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_PATHS = 2000

# The repository was already cloned into /workspace/repo by
# acquire_repository (Phase 21) — no clone happens here.
LOG_SCRIPT = """set -eu
cd /workspace/repo
git log --name-status --format='COMMIT %H%x1f%an%x1f%ae' HEAD
"""

SCOPE_SCRIPT = """set -eu
cd /workspace/repo
mkdir -p /workspace/scope
: > /workspace/scope.manifest
while IFS= read -r f; do
  [ -z "$f" ] && continue
  if [ -f "$f" ]; then
    mkdir -p "/workspace/scope/$(dirname "$f")"
    cp "$f" "/workspace/scope/$f"
    echo "OWNED $f"
    printf '%s\\n' "$f" >> /workspace/scope.manifest
  fi
done
echo "SCOPE_DONE"
"""


@dataclass(frozen=True)
class CommitChange:
    """One commit from the git history walk."""

    commit: str
    author_name: str
    author_email: str
    changes: list[str]


@dataclass(frozen=True)
class OwnershipResult:
    """Deterministic ownership computed from the git history."""

    owned_commits: list[str]
    current_paths: frozenset[str]
    deleted_paths: frozenset[str]
    renamed_from: dict[str, str]


def parse_git_log_output(output: str) -> list[CommitChange]:
    """Parse ``git log --name-status`` output into commit records.

    Each ``COMMIT <hash>\x1f<name>\x1f<email>`` line begins a record;
    the following tab-separated name-status lines belong to it.
    """
    changes: list[CommitChange] = []
    current: CommitChange | None = None
    for raw in output.splitlines():
        if not raw:
            continue
        if raw.startswith("COMMIT "):
            parts = raw[len("COMMIT "):].split("\x1f")
            if len(parts) != 3:
                current = None
                continue
            current = CommitChange(
                commit=parts[0],
                author_name=parts[1],
                author_email=parts[2],
                changes=[],
            )
            changes.append(current)
        elif current is not None:
            current.changes.append(raw)
    return changes


def commit_owned_by(change: CommitChange, contributor: str) -> bool:
    """Deterministic attribution of a commit to a GitHub login."""
    login = contributor.strip().lower()
    if not login:
        return False
    email = change.author_email.strip().lower()
    name = change.author_name.strip().lower()
    return (
        email == f"{login}@users.noreply.github.com"
        or email == f"{login}@github.com"
        or email.startswith(f"{login}@")
        or name == login
    )


def compute_ownership(
    changes: list[CommitChange], contributor: str
) -> OwnershipResult:
    """Collect paths touched by the contributor's commits.

    ``A/M/C/T`` contribute the path; ``R<score>`` contributes the rename
    destination (and records the source); ``D`` contributes nothing to
    the scope and is counted as deleted. Merge commits have no file
    lines and therefore contribute no paths.
    """
    owned_commits: list[str] = []
    current_paths: set[str] = set()
    deleted_paths: set[str] = set()
    renamed_from: dict[str, str] = {}

    for change in changes:
        if not commit_owned_by(change, contributor):
            continue
        owned_commits.append(change.commit)
        for line in change.changes:
            parts = line.split("\t")
            if not parts or not parts[0]:
                continue
            status = parts[0]
            if status.startswith("R"):
                if len(parts) >= 3:
                    renamed_from[parts[2]] = parts[1]
                    current_paths.add(parts[2])
            elif status == "D":
                if len(parts) >= 2:
                    deleted_paths.add(parts[1])
            elif len(parts) >= 2:
                current_paths.add(parts[1])

    return OwnershipResult(
        owned_commits=owned_commits,
        current_paths=frozenset(current_paths),
        deleted_paths=frozenset(deleted_paths),
        renamed_from=dict(renamed_from),
    )


class ContributorFilter:
    """Runs the deterministic ownership filter inside a Docker sandbox.

    Reuses the sandbox container registered by ``acquire_repository``
    (Phase 21): the repository is cloned once per pipeline and this tool
    only walks git history and builds the contributor scope.
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
    # Sandbox validation / cleanup
    # ------------------------------------------------------------------

    def _ensure_sandbox(self, container: str) -> None:
        ensure_sandbox_running(
            self._runner,
            container,
            timeout=DOCKER_INFO_TIMEOUT_SECONDS,
            context="filter outside a sandbox",
        )

    def _remove_container(self, container: str) -> None:
        try:
            self._runner([DOCKER_BIN, "rm", "-f", container], timeout=REMOVE_TIMEOUT_SECONDS)
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug("container %s already removed", container, exc_info=True)

    # ------------------------------------------------------------------
    # Sandbox operations
    # ------------------------------------------------------------------

    def _git_log(self, container: str) -> str:
        exec_cmd = [DOCKER_BIN, "exec", container, "/bin/sh", "-c", LOG_SCRIPT]
        try:
            result = self._runner(exec_cmd, timeout=FILTER_TIMEOUT_SECONDS)
        except _CommandTimeout:
            raise SandboxError(
                f"repository filtering timed out after {FILTER_TIMEOUT_SECONDS}s"
            ) from None
        if result.returncode != 0:
            raise SandboxError(
                f"git history walk failed inside the sandbox: {_sanitize_error(result.stderr, None)}"
            )
        if len(result.stdout) > MAX_FILTER_OUTPUT_BYTES:
            raise SandboxError("repository history output exceeds the size limit")
        return result.stdout

    def _build_scope(self, container: str, paths: set[str]) -> list[str]:
        sorted_paths = sorted(paths)
        exec_cmd = [DOCKER_BIN, "exec", "-i", container, "/bin/sh", "-c", SCOPE_SCRIPT]
        stdin_data = "\n".join(sorted_paths) + ("\n" if sorted_paths else "")
        try:
            result = self._runner(exec_cmd, timeout=SCOPE_TIMEOUT_SECONDS, stdin_data=stdin_data)
        except _CommandTimeout:
            raise SandboxError(f"scope build timed out after {SCOPE_TIMEOUT_SECONDS}s") from None
        if result.returncode != 0:
            raise SandboxError(
                f"scope build failed inside the sandbox: {_sanitize_error(result.stderr, None)}"
            )
        manifest: list[str] = []
        completed = False
        for line in result.stdout.splitlines():
            if line.startswith("OWNED "):
                manifest.append(line[len("OWNED "):])
            elif line == "SCOPE_DONE":
                completed = True
        if not completed:
            raise SandboxError("scope build did not complete")
        return sorted(set(manifest))

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_contributor_code(
        self,
        repository_identifier: str,
        contributor_identifier: str,
        sandbox_identifier: str,
    ) -> dict[str, object]:
        """Deterministically scope the repository to one contributor."""
        lease = self._registry.get(sandbox_identifier)
        if lease is None:
            raise SandboxError("sandbox unavailable: unknown sandbox_identifier")
        if lease.repository_identifier != repository_identifier:
            raise SandboxError(
                "sandbox unavailable: sandbox_identifier belongs to a different repository"
            )
        self._ensure_sandbox(sandbox_identifier)

        try:
            git_log_data = self._git_log(sandbox_identifier)
            changes = parse_git_log_output(git_log_data)
            ownership = compute_ownership(changes, contributor_identifier)
            manifest = self._build_scope(sandbox_identifier, ownership.current_paths)
            self._registry.attach_scope(
                sandbox_identifier,
                ContributorScope(
                    contributor_identifier=contributor_identifier,
                    workspace_path=SCOPE_WORKSPACE_PATH,
                    manifest=tuple(sorted(manifest)),
                ),
            )
            return self._report(
                repository_identifier,
                contributor_identifier,
                sandbox_identifier,
                changes,
                ownership,
                manifest,
            )
        except Exception:
            self._remove_container(sandbox_identifier)
            raise

    def _report(
        self,
        repository_identifier: str,
        contributor_identifier: str,
        sandbox_identifier: str,
        changes: list[CommitChange],
        ownership: OwnershipResult,
        manifest: list[str],
    ) -> dict[str, object]:
        total = len(changes)
        owned = len(ownership.owned_commits)
        deleted = len(ownership.deleted_paths)
        renamed = len(ownership.renamed_from)
        truncated = len(manifest) > MAX_MANIFEST_PATHS
        if owned == 0:
            status = "no_matching_commits"
        elif not manifest:
            status = "empty_scope"
        else:
            status = "filtered"
        return {
            "success": True,
            "repository_identifier": repository_identifier,
            "contributor_identifier": contributor_identifier,
            "sandbox_identifier": sandbox_identifier,
            "workspace_path": SCOPE_WORKSPACE_PATH,
            "status": status,
            "file_count": len(manifest),
            "owned_commit_count": owned,
            "total_commit_count": total,
            "deleted_file_count": deleted,
            "renamed_file_count": renamed,
            "manifest": list(manifest[:MAX_MANIFEST_PATHS]),
            "manifest_truncated": truncated,
            "message": f"contributor scope contains {len(manifest)} file(s)",
        }


__all__ = [
    "FILTER_TIMEOUT_SECONDS",
    "LOG_SCRIPT",
    "MAX_MANIFEST_PATHS",
    "SCOPE_SCRIPT",
    "CommitChange",
    "ContributorFilter",
    "OwnershipResult",
    "commit_owned_by",
    "compute_ownership",
    "parse_git_log_output",
]
