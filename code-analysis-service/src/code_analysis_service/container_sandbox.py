from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from .models import CloneResult

logger = logging.getLogger(__name__)

_IMAGE_TAG = "code-analysis-runner:latest"
_DOCKER_AVAILABLE: Optional[bool] = None

_REPO_DIR = "/workspace/repo"
_REPO_SIZE_LIMIT_BYTES = 500 * 1024 * 1024

# GNU coreutils `timeout` exit code when the wrapped command timed out.
_TIMED_OUT_EXIT_CODE = 124

# Exit code reported by the in-container process-group wrapper when a
# timed-out process group could NOT be positively confirmed terminated.
# The orchestrator treats any non-124 non-zero exit as a deterministic
# failure (tool_unavailable / clone_failed), i.e. NO retry — see
# `_PG_TIMEOUT_SH`.
_UNVERIFIED_EXIT_CODE = 125

# Margin added to the local `docker exec` client timeout: the in-container
# wrapper kills and VERIFIES the analyzer process group before it exits
# (124/125), so the local client is given extra time to receive that
# verified outcome. If the local timeout still fires, the process-group
# state is unknown and callers must fail closed (never retry).
_EXEC_TIMEOUT_MARGIN = 20.0

# Lease/TTL for stale-container protection (Finding 3): every analysis
# container and its dedicated network are labelled with an expiry
# timestamp at creation. A periodic sweep (see `stale_cleanup_loop`)
# force-removes any resource whose lease has expired — covering the case
# where the host process died mid-clone (finally never ran) and the
# keep-alive container would otherwise hold the PAT askpass indefinitely.
_LEASE_SECONDS = int(os.environ.get("CODE_ANALYSIS_CONTAINER_LEASE_SECONDS", "1800"))
_SWEEP_INTERVAL_SECONDS = int(
    os.environ.get("CODE_ANALYSIS_STALE_SWEEP_INTERVAL_SECONDS", "300")
)
_LEASE_LABEL = "code-analysis.lease-expires"

# In-container process-group timeout wrapper (Finding 2).
#
# The command runs as its own session/process-group leader (setsid); a
# watchdog SIGKILLs the ENTIRE group on expiry — the Python sandbox runner
# AND every analyzer descendant (eslint, pylint, radon, gocyclo,
# golangci-lint, rubocop, clippy, phpcs, checkstyle, dotnet, graphify,
# ...). Before returning, the wrapper POSITIVELY verifies the group is
# gone (kill -0 on the group id, i.e. the group exists iff any member
# remains). Exit codes:
#   124  command killed by the watchdog, group confirmed terminated
#        (transient timeout — same classification as GNU `timeout`)
#   125  termination could NOT be confirmed — FAIL CLOSED, no retry
#   else the command's own exit code, group confirmed empty
_PG_TIMEOUT_SH = """\
#!/bin/sh
# Usage: <timeout-seconds> <command...>
timeout_secs="$1"
shift
setsid "$@" &
child=$!
setsid sh -c 'sleep "$1"; kill -KILL -"$2"' watchdog "$timeout_secs" "$child" &
watcher=$!
wait "$child"
rc=$?
kill -KILL -"$watcher" 2>/dev/null
wait "$watcher" 2>/dev/null
i=0
while [ "$i" -lt 20 ] && kill -0 -"$child" 2>/dev/null; do
    sleep 0.1
    i=$((i + 1))
done
if kill -0 -"$child" 2>/dev/null; then
    kill -KILL -"$child" 2>/dev/null
    sleep 0.5
    if kill -0 -"$child" 2>/dev/null; then
        exit 125
    fi
fi
if [ "$rc" -gt 128 ]; then
    exit 124
fi
exit "$rc"
"""


def check_docker() -> bool:
    global _DOCKER_AVAILABLE
    if _DOCKER_AVAILABLE is not None:
        return _DOCKER_AVAILABLE
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=10,
        )
        _DOCKER_AVAILABLE = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        _DOCKER_AVAILABLE = False
    return _DOCKER_AVAILABLE


async def start_analysis_container(
    image: str = _IMAGE_TAG,
) -> tuple[str, str]:
    """Start ONE long-lived analysis container for a whole request.

    Returns ``(container_id, network_name)``. The container is created on
    a dedicated user-defined bridge network so the repository can be
    cloned; once the clone is complete the orchestrator calls
    ``disconnect_analysis_container``, leaving Static Analysis and
    Graphify with NO network access (loopback only).

    The container holds NO secrets: the PAT askpass script is written into
    the container's tmpfs ``/tmp`` immediately before each clone and
    deleted immediately afterwards (see ``clone_in_container``). No host
    files, no secret bind mounts, no secret environment variables.

    The container runs as the image's non-root ``analysis`` user; the
    writable directories (``/tmp``, ``/root``, ``/workspace``) are tmpfs
    mounts and ``HOME`` points at the writable ``/tmp``.

    The container and its dedicated network are labelled with a lease
    expiry timestamp; if the host process disappears mid-request the
    periodic stale sweep (``stale_cleanup_loop``) force-removes them once
    the lease expires, destroying any tmpfs that may still hold the PAT.
    """
    suffix = uuid.uuid4().hex[:12]
    name = f"code-analysis-{suffix}"
    network = f"code-analysis-net-{suffix}"
    lease_expiry = int(time.time()) + _LEASE_SECONDS
    container: Optional[str] = None
    network_created = False

    try:
        rc, stdout, stderr = await _docker_cli(
            [
                "docker", "network", "create",
                "--driver", "bridge",
                "--label", f"{_LEASE_LABEL}={lease_expiry}",
                network,
            ],
            timeout=30,
        )
        if rc != 0:
            raise RuntimeError(
                f"failed to create analysis network: {(stderr or stdout)[:300]}"
            )
        network_created = True

        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", name,
            "--network", network,
            "--label", f"{_LEASE_LABEL}={lease_expiry}",
            "--memory", "1024m",
            "--cpus", "2.0",
            "--read-only",
            "--tmpfs", "/tmp:rw,exec,nosuid,nodev",
            "--tmpfs", "/root:rw,exec,nosuid,nodev,uid=1000,gid=1000,mode=0700",
            "--tmpfs", "/workspace:rw,exec,nosuid,nodev,uid=1000,gid=1000,mode=0755",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "128",
            "-w", "/workspace",
            "-e", "HOME=/tmp",
            "-e", "GIT_TERMINAL_PROMPT=0",
            "-e", "DOTNET_CLI_TELEMETRY_OPTOUT=1",
            "-e", "DOTNET_NOLOGO=1",
            "-e", "DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1",
            image,
            "tail -f /dev/null",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(
                f"failed to start analysis container: {(stdout or stderr)[:500]}"
            )
        container = (stdout or b"").decode().strip()
        if not container:
            raise RuntimeError("docker did not return a container id")
        return container, network
    except asyncio.CancelledError:
        await _cleanup_partial(container, network, network_created)
        raise
    except Exception:
        await _cleanup_partial(container, network, network_created)
        raise


async def _cleanup_partial(
    container: Optional[str],
    network: Optional[str],
    network_created: bool,
) -> None:
    if container:
        remove_analysis_container(container, None)
    if network_created and network:
        remove_analysis_container(None, network)


async def exec_in_container(
    container: str,
    cmd: list[str],
    timeout: Optional[float] = None,
    env: Optional[list[str]] = None,
    input_bytes: Optional[bytes] = None,
) -> tuple[int, str, str]:
    """Run ``cmd`` inside a running container via ``docker exec``.

    Returns ``(returncode, stdout, stderr)``. ``env`` entries are
    ``KEY=VALUE`` pairs passed with ``-e`` (per-exec, not container-wide).
    ``input_bytes`` is written to the process stdin (used to create the
    askpass script without placing the PAT in argv or env).

    On timeout the local exec client is killed; the remote command is
    bounded by the in-container ``timeout`` wrapper added by the callers,
    so it cannot run indefinitely.
    """
    docker_cmd = ["docker", "exec"]
    if env:
        for kv in env:
            docker_cmd += ["-e", kv]
    if input_bytes is not None:
        docker_cmd.append("-i")
    docker_cmd += [container] + cmd
    try:
        proc = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdin=asyncio.subprocess.PIPE if input_bytes is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input_bytes), timeout=timeout
        )
    except asyncio.TimeoutError:
        _docker_kill(proc)
        raise
    except FileNotFoundError:
        raise RuntimeError("docker binary not found") from None
    return proc.returncode or 0, (stdout or b"").decode(), (stderr or b"").decode()


async def _docker_cli(
    cmd: list[str],
    timeout: float,
) -> tuple[int, str, str]:
    """Run a docker CLI command (network/container management)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        _docker_kill(proc)
        raise
    except FileNotFoundError:
        raise RuntimeError("docker binary not found") from None
    return proc.returncode or 0, (stdout or b"").decode(), (stderr or b"").decode()


async def clone_in_container(
    container: str,
    repo_url: str,
    pat: Optional[str],
    timeout_seconds: int,
    repo_dir: str = _REPO_DIR,
) -> CloneResult:
    """Clone ``repo_url`` into ``repo_dir`` inside the running container.

    When a PAT is supplied, the askpass script is written into the
    container's tmpfs ``/tmp`` immediately before the clone and deleted
    immediately afterwards — the PAT is never mounted, never placed in
    argv, never set as a container-wide env var, and never touches the
    host filesystem.

    Askpass cleanup is FAIL-CLOSED: if the deletion cannot be positively
    verified (``_remove_askpass_in_container``), the clone is reported as
    failed so the orchestrator NEVER starts Static Analysis or Graphify
    while the PAT might still exist inside the container. The clone is
    also bounded by the process-group timeout wrapper (``_PG_TIMEOUT_SH``)
    so a timed-out clone cannot leave a live ``git`` process behind.
    """
    await _exec_ignore(container, ["rm", "-rf", repo_dir])

    env: list[str] = []
    if pat:
        try:
            await _write_askpass_in_container(container, pat)
        except Exception as exc:
            return CloneResult(status="clone_failed", error_message=str(exc)[:500])
        env += ["GIT_ASKPASS=/tmp/askpass.sh", "GIT_TERMINAL_PROMPT=0"]

    failure_message: Optional[str] = None
    returncode = 0
    stdout = ""
    stderr = ""
    askpass_removed = True
    try:
        try:
            returncode, stdout, stderr = await exec_in_container(
                container,
                [
                    "sh", "-c", _PG_TIMEOUT_SH, "pg-timeout-wrapper",
                    str(timeout_seconds),
                    "git", "clone", "--depth=1", repo_url, repo_dir,
                ],
                timeout=float(timeout_seconds) + _EXEC_TIMEOUT_MARGIN,
                env=env,
            )
        except asyncio.TimeoutError:
            failure_message = "clone execution could not be verified"
        except Exception as exc:
            failure_message = str(exc)[:500]
    finally:
        if pat:
            askpass_removed = await _remove_askpass_in_container(container)

    if pat and not askpass_removed:
        logger.error(
            "PAT askpass script could not be confirmed removed after clone — "
            "failing closed; analysis will not start"
        )
        return CloneResult(
            status="clone_failed",
            error_message="askpass cleanup failed",
        )
    if failure_message is not None:
        return CloneResult(status="clone_failed", error_message=failure_message)

    output = stdout + stderr
    if returncode == _TIMED_OUT_EXIT_CODE:
        return CloneResult(status="clone_failed", error_message="clone timed out")
    if returncode == _UNVERIFIED_EXIT_CODE:
        return CloneResult(
            status="clone_failed",
            error_message="clone process group termination could not be verified",
        )
    if returncode != 0:
        lower = output.lower()
        if "authentication failed" in lower or "access denied" in lower:
            return CloneResult(status="authentication_failed", error_message=output)
        if "repository not found" in lower:
            return CloneResult(status="clone_failed", error_message=output)
        return CloneResult(status="clone_failed", error_message=output[:500])

    repo_size = await _repo_size(container, repo_dir)
    if repo_size > _REPO_SIZE_LIMIT_BYTES:
        await _exec_ignore(container, ["rm", "-rf", repo_dir])
        return CloneResult(status="repo_too_large")
    return CloneResult(status="success", repo_path=repo_dir)


async def disconnect_analysis_container(container: str, network: str) -> bool:
    """Remove the container from its network so analysis has NO network access.

    Returns True when the container is left with loopback only. A
    disconnect failure (or an already-disconnected container) is handled
    so the call is idempotent; on a real failure False is returned and the
    caller must fail closed rather than analyze with network access.
    """
    try:
        rc, _stdout, stderr = await _docker_cli(
            ["docker", "network", "disconnect", network, container],
            timeout=30,
        )
    except Exception as exc:
        logger.warning("failed to disconnect analysis container: %s", exc)
        return False
    if rc != 0 and "is not connected" not in stderr.lower():
        logger.warning("failed to disconnect analysis container: %s", stderr[:300])
        return False
    return True


async def run_analysis_in_container(
    container: str,
    tool: str,
    language: str,
    repo_dir: str,
    timeout: Optional[float] = None,
) -> tuple[int, str, str]:
    """Run a static-analysis/Graphify tool via ``docker exec``.

    The command is wrapped in the process-group timeout wrapper
    (``_PG_TIMEOUT_SH``), so a timed-out analysis kills the ENTIRE
    analyzer process group — the Python runner AND every analyzer child
    (eslint, pylint, radon, gocyclo, golangci-lint, rubocop, clippy,
    phpcs, checkstyle, dotnet, graphify, ...). Exit code 124 signals a
    timeout whose process group was verified terminated; exit code 125
    means termination could not be confirmed (fail closed, no retry).

    Static Analysis additionally gets ``NODE_PATH`` pointing at the
    globally installed Node modules so ESLint can resolve the
    TypeScript parser for ``.ts``/``.tsx`` files.
    """
    if tool == "static_analysis":
        runner_cmd = [
            "python", "-m", "code_analysis_service.sandbox_runner",
            "static", repo_dir,
            "--language", language,
        ]
        env = ["NODE_PATH=/usr/local/lib/node_modules"]
    elif tool == "graph_relations":
        runner_cmd = [
            "python", "-m", "code_analysis_service.sandbox_runner",
            "graph", repo_dir,
        ]
        env = []
    else:
        raise ValueError(f"unsupported sandbox tool: {tool}")

    exec_timeout: Optional[float] = None
    if timeout is not None:
        runner_cmd = (
            ["sh", "-c", _PG_TIMEOUT_SH, "pg-timeout-wrapper", str(int(timeout))]
            + runner_cmd
        )
        # The in-container wrapper kills and VERIFIES the whole process
        # group before exiting (124/125); the local docker exec client gets
        # a margin so the verified outcome arrives first. If the local
        # timeout still fires, the group state is unknown and the caller
        # must fail closed (no retry).
        exec_timeout = float(timeout) + _EXEC_TIMEOUT_MARGIN

    return await exec_in_container(container, runner_cmd, timeout=exec_timeout, env=env)


async def is_workspace_available(container: str) -> bool:
    """True if the container still holds the cloned repo at ``/workspace/repo``."""
    try:
        returncode, _stdout, _stderr = await exec_in_container(
            container, ["test", "-d", _REPO_DIR], timeout=30
        )
    except Exception:
        return False
    return returncode == 0


def remove_analysis_container(
    container: Optional[str],
    network: Optional[str] = None,
) -> None:
    """Best-effort cleanup of the container and its dedicated network. Never raises."""
    if container:
        try:
            subprocess.run(
                ["docker", "rm", "-f", container],
                capture_output=True, text=True, timeout=60,
            )
        except Exception:
            logger.warning("failed to remove analysis container %s", container)
    if network:
        try:
            subprocess.run(
                ["docker", "network", "rm", network],
                capture_output=True, text=True, timeout=60,
            )
        except Exception:
            logger.warning("failed to remove analysis network %s", network)


async def stale_cleanup_loop() -> None:
    """Periodically remove analysis containers/networks whose lease expired.

    Every analysis container and its dedicated network carry a lease
    expiry label (``_LEASE_LABEL``), set at creation. If the host process
    dies mid-request (crash, SIGKILL), the orchestrator's ``finally``
    never runs and the keep-alive container would otherwise live forever —
    still on its network and possibly still holding the PAT askpass in its
    tmpfs. This loop runs one sweep immediately (covering orphans from a
    previous crash) and then every ``_SWEEP_INTERVAL_SECONDS``; expired
    resources are force-removed together with their dedicated network,
    destroying the tmpfs that may contain the PAT.
    """
    while True:
        try:
            await _reap_stale_analysis_resources()
        except Exception:
            logger.exception("stale analysis resource sweep failed")
        await asyncio.sleep(_SWEEP_INTERVAL_SECONDS)


async def _reap_stale_analysis_resources() -> None:
    """Best-effort removal of lease-expired analysis containers and networks."""
    now = int(time.time())
    try:
        _rc, stdout, stderr = await _docker_cli(
            [
                "docker", "ps", "-a",
                "--filter", f"label={_LEASE_LABEL}",
                "--format", f"{{{{.ID}}}} {{{{.Label \"{_LEASE_LABEL}\"}}}}",
            ],
            timeout=30,
        )
        if _rc != 0:
            logger.warning(
                "stale sweep: cannot list analysis containers: %s",
                (stderr or "")[:200],
            )
        else:
            for line in (stdout or "").splitlines():
                parts = line.split()
                if len(parts) != 2 or not parts[1].isdigit():
                    continue
                cid, expiry = parts[0], parts[1]
                if int(expiry) < now:
                    logger.warning(
                        "stale sweep: removing lease-expired container %s", cid
                    )
                    remove_analysis_container(cid)
    except Exception as exc:
        logger.warning("stale sweep: container scan failed: %s", exc)

    try:
        _rc, stdout, stderr = await _docker_cli(
            [
                "docker", "network", "ls",
                "--filter", f"label={_LEASE_LABEL}",
                "--format", f"{{{{.ID}}}} {{{{.Label \"{_LEASE_LABEL}\"}}}}",
            ],
            timeout=30,
        )
        if _rc != 0:
            logger.warning(
                "stale sweep: cannot list analysis networks: %s",
                (stderr or "")[:200],
            )
        else:
            for line in (stdout or "").splitlines():
                parts = line.split()
                if len(parts) != 2 or not parts[1].isdigit():
                    continue
                nid, expiry = parts[0], parts[1]
                if int(expiry) < now:
                    logger.warning(
                        "stale sweep: removing lease-expired network %s", nid
                    )
                    remove_analysis_container(None, nid)
    except Exception as exc:
        logger.warning("stale sweep: network scan failed: %s", exc)


async def _write_askpass_in_container(container: str, pat: str) -> None:
    """Write the PAT askpass script into the container's tmpfs /tmp.

    The PAT travels over the exec stdin only — never through argv, env or
    a host file. The script is removed right after the clone.
    """
    import shlex
    content = f"#!/bin/sh\necho {shlex.quote(pat)}\n".encode()
    rc, _stdout, stderr = await exec_in_container(
        container,
        ["sh", "-c", "umask 077; cat > /tmp/askpass.sh && chmod 700 /tmp/askpass.sh"],
        timeout=15,
        input_bytes=content,
    )
    if rc != 0:
        raise RuntimeError(f"failed to write askpass in container: {stderr[:300]}")


async def _remove_askpass_in_container(container: str) -> bool:
    """Remove the PAT askpass script and positively verify its deletion.

    Returns True only when ``/tmp/askpass.sh`` is confirmed gone. On any
    failure (command error, exec failure, verification failure) the caller
    must FAIL CLOSED: Static Analysis and Graphify must never start while
    the PAT might still exist inside the container. The PAT itself is
    never logged or exposed in errors.
    """
    try:
        rc, _stdout, _stderr = await exec_in_container(
            container, ["rm", "-f", "/tmp/askpass.sh"], timeout=15
        )
    except Exception as exc:
        logger.warning("askpass removal command failed: %s", exc)
        return False
    if rc != 0:
        logger.warning("askpass removal command exited %s", rc)
        return False
    try:
        rc, _stdout, _stderr = await exec_in_container(
            container, ["test", "!", "-e", "/tmp/askpass.sh"], timeout=15
        )
    except Exception as exc:
        logger.warning("askpass removal verification failed: %s", exc)
        return False
    if rc != 0:
        logger.warning("askpass script still present after removal attempt")
        return False
    return True


async def _repo_size(container: str, repo_dir: str) -> int:
    try:
        _rc, stdout, _stderr = await exec_in_container(
            container, ["du", "-sb", repo_dir], timeout=60
        )
    except Exception:
        return 0
    try:
        return int(stdout.split()[0])
    except (ValueError, IndexError):
        return 0


async def _exec_ignore(container: str, cmd: list[str]) -> None:
    try:
        await exec_in_container(container, cmd, timeout=30)
    except Exception:
        pass


def _docker_kill(proc: asyncio.subprocess.Process) -> None:
    try:
        proc.kill()
    except ProcessLookupError:
        pass


@asynccontextmanager
async def sandbox_context(
    disk_limit_mb: int = 500,
    cpu_limit: float = 2.0,
    memory_limit_mb: int = 1024,
    network_enabled: bool = False,
) -> AsyncIterator["SandboxWorkspace"]:
    tmp = tempfile.mkdtemp(prefix="code-analysis-container-")
    try:
        yield SandboxWorkspace(
            work_dir=Path(tmp),
            memory_limit_mb=memory_limit_mb,
            cpu_limit=cpu_limit,
            disk_limit_mb=disk_limit_mb,
            network_enabled=network_enabled,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class SandboxWorkspace:
    def __init__(
        self,
        work_dir: Path,
        memory_limit_mb: int,
        cpu_limit: float,
        disk_limit_mb: int,
        network_enabled: bool,
    ):
        self.work_dir = work_dir
        self._memory = memory_limit_mb
        self._cpu = cpu_limit
        self._disk = disk_limit_mb
        self._network = network_enabled