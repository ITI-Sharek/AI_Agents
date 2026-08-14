"""Docker-sandboxed repository acquisition.

Phase 4: clones an untrusted repository inside an isolated, ephemeral
Docker container. The clone never touches the host filesystem and no
host-side git operations are used.

Phase 21: the sandbox container is now PERSISTENT per acquired
repository. ``acquire_repository`` starts one disposable container that
runs the analysis image (which bundles git, the static-analysis tools,
and Graphify), clones the FULL repository inside it, and keeps the
container running so ``filter_contributor_code``, ``analyze_static`` and
``analyze_graph`` can reuse the exact same repository workspace without
re-cloning or rebuilding the scope. The container is removed on failure
and orphaned containers (not referenced by any registered lease) are
swept on the next acquisition.

The repository URL and the GitHub PAT are streamed into the container via
stdin: the URL and PAT never appear on the container command line, in the
container configuration, in logs, or in error messages. The PAT is only
used for the single authenticated clone and is scrubbed from the
container afterwards.

Phase 22: the analysis image is verified for FRESHNESS, not just
presence — the image carries a content fingerprint of its build inputs
(see ``IMAGE_SOURCE_LABEL``) and a stale image is rebuilt before use.

Phase 24: the sandbox lifecycle is bounded. ``acquire`` registers the
container as a TTL-bounded lease (see ``skill_profiling_mcp.registry``)
BEFORE cloning, while holding the registry lock across the orphan
sweep, container start, and registration — a concurrent request can
never sweep a just-started, not-yet-registered container. On failure the
container and its lease are removed immediately; on success the lease
expires after ``SANDBOX_TTL_SECONDS`` and the sweep on the next
acquisition reaps the container.
"""

import hashlib
import logging
import re
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from skill_profiling_mcp.security import redact, safe_branch_name

logger = logging.getLogger(__name__)

DOCKER_BIN = "docker"
SANDBOX_IMAGE = "alpine/git"
ANALYSIS_IMAGE = "skill-profiling-mcp-analysis:latest"
DOCKERFILE_NAME = "Dockerfile.analysis"

# Docker label stamped on the analysis image with the content fingerprint
# of its build inputs (the Dockerfile and the package sources copied into
# the image). ``_ensure_image`` compares the label against the current
# sources: an image built from older code has a different fingerprint and
# is rebuilt before use.
IMAGE_SOURCE_LABEL = "com.skill-profiling.analysis.source-sha256"

MEMORY_LIMIT = "512m"
CPU_LIMIT = "1.0"
PIDS_LIMIT = 128
CLONE_TIMEOUT_SECONDS = 300
DOCKER_INFO_TIMEOUT_SECONDS = 15
REMOVE_TIMEOUT_SECONDS = 30
IMAGE_INSPECT_TIMEOUT_SECONDS = 15
IMAGE_BUILD_TIMEOUT_SECONDS = 600
MAX_ERROR_OUTPUT_CHARS = 2000

KEEPALIVE_SCRIPT = "tail -f /dev/null"

# Containers created by this system are named skill-profiling-mcp-*.
_SANDBOX_NAME_RE = re.compile(r"^skill-profiling-mcp-[0-9a-f]{12}$")

# Whether the analysis image was verified/built in this process for the
# current source fingerprint. The fingerprint is recomputed on every call
# (cheap: only the runner sources are hashed), so editing the analysis
# code invalidates the cached check and forces a fresh ``docker image
# inspect`` — a stale image is never silently reused. If `docker run`
# reports the image missing the cache is cleared and the image rebuilt.
_IMAGE_ENSURE_FLAG = False
_IMAGE_ENSURE_FINGERPRINT: str | None = None

CLONE_SCRIPT = """set -eu
read REPO_URL || exit 2
read GITHUB_PAT || GITHUB_PAT=
export GIT_TERMINAL_PROMPT=0
ASKPASS=/tmp/git-askpass
printf '#!/bin/sh\\ncase "$1" in\\n  Username*) echo "x-access-token" ;;\\n  Password*) printf "%s" "$GITHUB_PAT" ;;\\nesac\\n' > "$ASKPASS"
chmod 700 "$ASKPASS"
export GIT_ASKPASS="$ASKPASS"
mkdir -p /workspace
cd /workspace
git clone --quiet "$REPO_URL" repo
BRANCH=$(git -C /workspace/repo rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
rm -f "$ASKPASS"
unset GITHUB_PAT GIT_ASKPASS
echo "ACQUIRE_OK"
echo "BRANCH=$BRANCH"
echo "WORKSPACE=/workspace/repo"
"""


class DockerUnavailableError(RuntimeError):
    """Raised when the Docker daemon is not reachable (fail closed)."""


class SandboxError(RuntimeError):
    """Raised when sandbox creation, cloning, or cleanup fails."""


class _CommandTimeout(RuntimeError):
    """Internal: a docker command exceeded its time limit."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def _run_command(cmd: list[str], *, timeout: int, stdin_data: str | None = None) -> CommandResult:
    """Execute a fixed, host-side docker CLI command (no shell, exec array)."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            input=stdin_data,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise _CommandTimeout(timeout) from exc
    return CommandResult(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def _sanitize_error(text: str | None, secret: str | None) -> str:
    if not text:
        return "unknown error"
    text = text.strip()
    if len(text) > MAX_ERROR_OUTPUT_CHARS:
        text = text[-MAX_ERROR_OUTPUT_CHARS:]
    return redact(secret, text)


def _repo_root() -> Path:
    """Repository root derived from this module's location."""
    return Path(__file__).resolve().parents[2]


def _analysis_image_files() -> tuple[Path, Path]:
    """Return ``(dockerfile, build_context)`` for the analysis image."""
    root = _repo_root()
    dockerfile = root / "skill_profiling_mcp" / "docker" / DOCKERFILE_NAME
    return dockerfile, root


def _image_build_inputs() -> list[Path]:
    """Return the files whose contents determine the analysis image.

    The Dockerfile, everything bundled from the ``skill_profiling_mcp``
    docker directory (including the in-image TypeScript ESLint config and
    the offline Semgrep ruleset), and the ``skill_profiling_mcp`` package
    sources that the Dockerfile copies into the image (``COPY
    skill_profiling_mcp /opt/skill-profiling-mcp``). Cache artifacts are
    excluded. A change to any of these inputs must produce a different
    image.
    """
    dockerfile, _ = _analysis_image_files()
    package = _repo_root() / "skill_profiling_mcp"
    inputs = [dockerfile]
    docker_dir = package / "docker"
    if docker_dir.is_dir():
        for path in sorted(docker_dir.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                inputs.append(path)
    for path in sorted(package.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        inputs.append(path)
    return inputs


def _source_fingerprint() -> str:
    """Content hash of the analysis image build inputs.

    Paths (repo-relative) and bytes of every build input are hashed, so
    renamed or moved sources also invalidate the fingerprint. A read
    failure fails closed: the image cannot be verified fresh.
    """
    hasher = hashlib.sha256()
    root = _repo_root()
    try:
        for path in _image_build_inputs():
            hasher.update(str(path.relative_to(root)).encode("utf-8"))
            hasher.update(path.read_bytes())
    except OSError as exc:
        raise SandboxError(f"cannot fingerprint analysis image sources: {exc}") from exc
    return hasher.hexdigest()


def reset_image_ensure_cache() -> None:
    """Clear the in-process analysis-image verification cache (test hook)."""
    global _IMAGE_ENSURE_FLAG, _IMAGE_ENSURE_FINGERPRINT
    _IMAGE_ENSURE_FLAG = False
    _IMAGE_ENSURE_FINGERPRINT = None


def ensure_sandbox_running(
    runner: Callable[..., CommandResult],
    container: str,
    *,
    timeout: int,
    context: str,
) -> None:
    """Validate the registered sandbox container (fail closed).

    Runs a single ``docker inspect`` as the per-call gate for the tools
    that REUSE the sandbox container: when the daemon is unreachable the
    call fails with ``DockerUnavailableError``; when the container is
    missing or no longer running it fails with ``SandboxError`` (the
    caller must re-acquire the repository). No analysis ever runs outside
    the sandbox.
    """
    inspect = runner(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        timeout=timeout,
    )
    if inspect.returncode == 0:
        if inspect.stdout.strip() == "true":
            return
        raise SandboxError(
            "sandbox unavailable: sandbox container is not running; "
            "re-acquire the repository"
        )
    info = runner(["docker", "info"], timeout=timeout)
    if info.returncode != 0:
        raise DockerUnavailableError(
            f"Docker is unavailable; refusing to {context} outside a sandbox"
        )
    raise SandboxError(
        "sandbox unavailable: sandbox container is missing; re-acquire the repository"
    )


def _ensure_image(
    runner: Callable[..., CommandResult],
    image: str = ANALYSIS_IMAGE,
) -> None:
    """Require a FRESH analysis image; build only when it is missing or stale.

    The image is considered current only when its ``IMAGE_SOURCE_LABEL``
    equals the content fingerprint of the current build inputs. A stale
    image (built from older ``analysis_runner.py`` / runner sources) is
    rebuilt before use, so changed analysis code always results in a
    current image — never a silent reuse of an outdated one.

    The verification is cached per process for the CURRENT fingerprint:
    repeated calls with unchanged sources only hash the inputs (no docker
    calls), so there is no unnecessary rebuild and no repeated inspect on
    every request. When the fingerprint changes (analysis code edited) or
    `docker run` reports the image missing (see
    ``RepositoryAcquirer._start_container``) the cache is revalidated.
    """
    global _IMAGE_ENSURE_FLAG, _IMAGE_ENSURE_FINGERPRINT
    fingerprint = _source_fingerprint()
    if _IMAGE_ENSURE_FLAG and _IMAGE_ENSURE_FINGERPRINT == fingerprint:
        return
    inspect = runner(
        [
            "docker",
            "image",
            "inspect",
            "-f",
            f'{{{{ index .Config.Labels "{IMAGE_SOURCE_LABEL}" }}}}',
            image,
        ],
        timeout=IMAGE_INSPECT_TIMEOUT_SECONDS,
    )
    if inspect.returncode == 0 and inspect.stdout.strip() == fingerprint:
        _IMAGE_ENSURE_FLAG = True
        _IMAGE_ENSURE_FINGERPRINT = fingerprint
        return
    dockerfile, context = _analysis_image_files()
    build = runner(
        [
            DOCKER_BIN,
            "build",
            "-t",
            image,
            "-f",
            str(dockerfile),
            "--label",
            f"{IMAGE_SOURCE_LABEL}={fingerprint}",
            str(context),
        ],
        timeout=IMAGE_BUILD_TIMEOUT_SECONDS,
    )
    if build.returncode != 0:
        raise SandboxError(
            f"analysis sandbox image build failed: {_sanitize_error(build.stderr, None)}"
        )
    _IMAGE_ENSURE_FLAG = True
    _IMAGE_ENSURE_FINGERPRINT = fingerprint


def _container_name() -> str:
    return f"skill-profiling-mcp-{uuid.uuid4().hex[:12]}"


class RepositoryAcquirer:
    """Acquires a repository inside a disposable Docker sandbox.

    The sandbox container stays running after acquisition so the
    contributor filter and both analyzers can reuse the cloned
    repository workspace (Phase 21). It is removed on failure, and
    orphaned sandbox containers are swept on the next acquisition.
    """

    def __init__(
        self,
        *,
        runner: Callable[..., CommandResult] = _run_command,
        image: str = ANALYSIS_IMAGE,
    ) -> None:
        self._runner = runner
        self._image = image

    # ------------------------------------------------------------------
    # Docker daemon check (fail closed, never host-side clone)
    # ------------------------------------------------------------------

    def _docker_available(self) -> bool:
        result = self._runner(
            [DOCKER_BIN, "info"],
            timeout=DOCKER_INFO_TIMEOUT_SECONDS,
        )
        return result.returncode == 0

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def _remove_container(self, container: str) -> None:
        try:
            self._runner([DOCKER_BIN, "rm", "-f", container], timeout=REMOVE_TIMEOUT_SECONDS)
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug("container %s already removed", container, exc_info=True)

    def _start_container(self, container: str) -> None:
        run_cmd = [
            DOCKER_BIN,
            "run",
            "-d",
            "--name", container,
            "--memory", MEMORY_LIMIT,
            "--memory-swap", MEMORY_LIMIT,
            "--cpus", CPU_LIMIT,
            "--pids-limit", str(PIDS_LIMIT),
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only",
            "--tmpfs", "/tmp",
            "--tmpfs", "/root",
            "--tmpfs", "/workspace",
            "--entrypoint", "/bin/sh",
            self._image,
            "-c",
            KEEPALIVE_SCRIPT,
        ]
        result = self._runner(run_cmd, timeout=REMOVE_TIMEOUT_SECONDS)
        if result.returncode != 0:
            if "Unable to find image" in (result.stderr or ""):
                reset_image_ensure_cache()
            raise SandboxError(
                f"sandbox creation failed: {_sanitize_error(result.stderr, None)}"
            )

    def _sweep_orphan_sandboxes(self) -> None:
        """Remove sandbox-named containers that no live lease references.

        A container is removed when either no registered lease references
        it (orphaned — e.g. after a server restart the in-memory registry
        is lost) or its registered lease has EXPIRED (the sandbox TTL
        bounds every container's lifetime). Expired leases are pruned
        from the registry at the same time. Never touches containers
        referenced by live, unexpired leases. Best effort: failures are
        logged, not fatal. The caller must hold the registry lock
        (``sandbox_registry.locked()``).
        """
        from skill_profiling_mcp.registry import sandbox_registry

        registered = set(sandbox_registry.sandbox_ids())
        expired = set(sandbox_registry.expired_ids())
        for sandbox_identifier in expired:
            sandbox_registry.remove(sandbox_identifier)
        try:
            result = self._runner(
                [DOCKER_BIN, "ps", "-a", "--no-trunc", "--format", "{{.Names}}"],
                timeout=REMOVE_TIMEOUT_SECONDS,
            )
        except Exception:  # pragma: no cover - best effort sweep
            logger.debug("orphan sandbox sweep failed", exc_info=True)
            return
        for name in result.stdout.split():
            if _SANDBOX_NAME_RE.fullmatch(name) and (name not in registered or name in expired):
                logger.debug("removing expired/orphaned sandbox container %s", name)
                self._remove_container(name)

    def dispose(self, sandbox_identifier: str) -> None:
        """Remove a sandbox's lease and container (best effort, idempotent).

        Used by the orchestration tool's failure path so a failed
        analysis request never leaves a registered lease or a running
        container behind. Docker failures are logged, never raised.
        """
        from skill_profiling_mcp.registry import sandbox_registry

        sandbox_registry.remove(sandbox_identifier)
        self._remove_container(sandbox_identifier)

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def acquire(self, repo_url: str, identifier: str, github_pat: str | None = None) -> dict[str, object]:
        """Clone `repo_url` inside an isolated Docker sandbox container.

        The sandbox container is kept running after the clone so later
        tools (filter / static / graph) reuse the same repository
        workspace. The container is registered as a TTL-bounded lease
        (see ``skill_profiling_mcp.registry``) BEFORE the clone, while
        holding the registry lock across the orphan sweep, container
        start, and registration — a concurrent request can never sweep
        this just-started container. On failure the container and its
        lease are removed immediately. Returns safe metadata only. Never
        returns the PAT or repository contents. Raises
        DockerUnavailableError or SandboxError on failure.
        """
        if not self._docker_available():
            raise DockerUnavailableError(
                "Docker is unavailable; refusing to acquire the repository outside a sandbox"
            )
        _ensure_image(self._runner, self._image)

        from skill_profiling_mcp.registry import SandboxLease, sandbox_registry

        with sandbox_registry.locked():
            self._sweep_orphan_sandboxes()
            container = _container_name()
            self._start_container(container)
            sandbox_registry.register(
                SandboxLease(
                    sandbox_identifier=container,
                    repository_identifier=identifier,
                    repo_url=repo_url,
                    image=self._image,
                )
            )

        keep = False
        try:
            stdin_data = repo_url + "\n" + (github_pat + "\n" if github_pat else "")

            exec_cmd = [
                DOCKER_BIN,
                "exec",
                "-i",
                container,
                "/bin/sh",
                "-c",
                CLONE_SCRIPT,
            ]

            logger.info("acquiring repository %s in sandbox container %s", identifier, container)
            try:
                result = self._runner(exec_cmd, timeout=CLONE_TIMEOUT_SECONDS, stdin_data=stdin_data)
            except _CommandTimeout:
                raise SandboxError(
                    f"repository acquisition timed out after {CLONE_TIMEOUT_SECONDS}s"
                ) from None

            acquired = self._interpret(identifier, container, github_pat, result)
            keep = True
            return acquired
        finally:
            if not keep:
                self._remove_container(container)
                sandbox_registry.remove(container)

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    def _interpret(
        self,
        identifier: str,
        container: str,
        github_pat: str | None,
        result: CommandResult,
    ) -> dict[str, object]:
        if result.returncode != 0:
            detail = _sanitize_error(result.stderr or result.stdout, github_pat)
            raise SandboxError(f"clone failed inside the sandbox: {detail}")

        acquired = False
        branch: str | None = None
        workspace: str | None = None
        for line in result.stdout.splitlines():
            if line == "ACQUIRE_OK":
                acquired = True
            elif line.startswith("BRANCH="):
                branch = line[len("BRANCH="):]
            elif line.startswith("WORKSPACE="):
                workspace = line[len("WORKSPACE="):]

        if not acquired:
            detail = _sanitize_error(result.stderr, github_pat)
            raise SandboxError(f"repository acquisition did not complete: {detail}")

        return {
            "success": True,
            "repository_identifier": identifier,
            "default_branch": safe_branch_name(branch),
            "sandbox_identifier": container,
            "workspace_path": workspace or "/workspace/repo",
            "message": "repository acquired inside an isolated Docker sandbox",
        }


__all__ = [
    "ANALYSIS_IMAGE",
    "CLONE_SCRIPT",
    "CLONE_TIMEOUT_SECONDS",
    "CPU_LIMIT",
    "DOCKERFILE_NAME",
    "DOCKER_INFO_TIMEOUT_SECONDS",
    "IMAGE_BUILD_TIMEOUT_SECONDS",
    "IMAGE_INSPECT_TIMEOUT_SECONDS",
    "IMAGE_SOURCE_LABEL",
    "KEEPALIVE_SCRIPT",
    "MEMORY_LIMIT",
    "PIDS_LIMIT",
    "CommandResult",
    "DockerUnavailableError",
    "RepositoryAcquirer",
    "SandboxError",
    "ensure_sandbox_running",
    "reset_image_ensure_cache",
]
