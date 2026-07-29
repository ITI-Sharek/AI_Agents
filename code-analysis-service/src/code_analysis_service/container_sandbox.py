from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from .models import CloneResult

logger = logging.getLogger(__name__)

_IMAGE_TAG = "code-analysis-runner:latest"
_DOCKER_AVAILABLE: Optional[bool] = None


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


async def clone_in_container(
    repo_url: str,
    pat: Optional[str],
    dest_dir: str,
    timeout_seconds: int,
    image: str = _IMAGE_TAG,
) -> CloneResult:
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    env_args: list[str] = []
    if pat:
        askpass_script = _create_askpass(pat)
        env_args = [
            "-e", f"GIT_ASKPASS=/tmp/askpass.sh",
            "-v", f"{askpass_script}:/tmp/askpass.sh:ro",
        ]

    cmd = [
        "docker", "run", "--rm",
        "--memory", "1024m",
        "--cpus", "2.0",
        "--network", "bridge",
        "-v", f"{dest_dir}:/workspace",
        "-w", "/workspace",
    ] + env_args + [
        image,
        "sh", "-c",
        f"git clone --depth=1 {_quote(repo_url)} /workspace/repo 2>&1",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        _docker_kill(proc)
        return CloneResult(status="clone_failed", error_message="clone timed out")
    except FileNotFoundError:
        return CloneResult(status="clone_failed", error_message="docker binary not found")

    output = (stdout or b"").decode() + (stderr or b"").decode()

    if proc.returncode != 0:
        lower = output.lower()
        if "authentication failed" in lower or "access denied" in lower:
            return CloneResult(status="authentication_failed", error_message=output)
        if "repository not found" in lower:
            return CloneResult(status="clone_failed", error_message=output)
        return CloneResult(status="clone_failed", error_message=output[:500])

    repo_path = dest_path / "repo"
    if not repo_path.exists():
        return CloneResult(status="clone_failed", error_message="clone did not produce repo directory")

    repo_size = _dir_size(str(repo_path))
    if repo_size > 500 * 1024 * 1024:
        shutil.rmtree(str(repo_path), ignore_errors=True)
        return CloneResult(status="repo_too_large")

    _move_up(repo_path, dest_path)
    return CloneResult(status="success", repo_path=str(dest_path))


def _create_askpass(pat: str) -> str:
    import shlex
    quoted = shlex.quote(pat)
    content = f"#!/bin/sh\necho {quoted}\n"
    fd, path = tempfile.mkstemp(suffix=".sh", prefix="git-askpass-")
    os.close(fd)
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, 0o700)
    return path


def _quote(s: str) -> str:
    return s.replace("'", "'\\''")


def _dir_size(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def _move_up(src: Path, dest: Path) -> None:
    for item in src.iterdir():
        shutil.move(str(item), str(dest / item.name))
    src.rmdir()


def _docker_kill(proc: asyncio.subprocess.Process) -> None:
    try:
        proc.kill()
    except ProcessLookupError:
        pass


async def run_in_container(
    cmd: list[str],
    work_dir: str,
    timeout: Optional[float] = None,
    memory_mb: int = 1024,
    cpus: float = 2.0,
    network: bool = False,
) -> tuple[int, str, str]:
    docker_cmd = [
        "docker", "run", "--rm",
        "--memory", f"{memory_mb}m",
        "--cpus", str(cpus),
        "--network", "bridge" if network else "none",
        "-v", f"{work_dir}:/workspace:ro",
        "-w", "/workspace",
        _IMAGE_TAG,
    ] + cmd

    try:
        proc = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        return proc.returncode or 0, (stdout or b"").decode(), (stderr or b"").decode()
    except asyncio.TimeoutError:
        _docker_kill(proc)
        raise


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
