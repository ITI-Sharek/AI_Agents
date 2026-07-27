from __future__ import annotations

import asyncio
import logging
import os
import shlex
import tempfile
from pathlib import Path
from typing import Optional

from .models import CloneResult

logger = logging.getLogger(__name__)


def _build_askpass_script(pat: str) -> tuple[str, str]:
    quoted = shlex.quote(pat)
    content = f"#!/bin/sh\necho {quoted}\n"
    fd, path = tempfile.mkstemp(suffix=".sh", prefix="git-askpass-")
    os.close(fd)
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, 0o700)
    return path, content


async def clone_repo(
    repo_url: str,
    pat: Optional[str],
    dest_dir: str,
    timeout_seconds: int,
    disk_limit_bytes: int = 0,
) -> CloneResult:
    dest_path = Path(dest_dir)

    if pat is None:
        cmd = ["git", "clone", "--depth=1", repo_url, str(dest_path)]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            return CloneResult(
                status="clone_failed", error_message="clone timed out"
            )
        except FileNotFoundError:
            return CloneResult(
                status="clone_failed", error_message="git binary not found"
            )

        stderr_text = stderr.decode() if stderr else ""
        if proc.returncode != 0:
            if "repository not found" in stderr_text.lower():
                return CloneResult(
                    status="clone_failed", error_message=stderr_text
                )
            return CloneResult(
                status="clone_failed", error_message=stderr_text
            )

        if (
            disk_limit_bytes > 0
            and _dir_size(str(dest_path)) > disk_limit_bytes
        ):
            _rmtree(str(dest_path))
            return CloneResult(status="repo_too_large")

        return CloneResult(status="success", repo_path=str(dest_path))

    script_path: Optional[str] = None
    try:
        askpass_path, _script_content = _build_askpass_script(pat)
        script_path = askpass_path

        env = os.environ.copy()
        env["GIT_ASKPASS"] = askpass_path

        cmd = ["git", "clone", "--depth=1", repo_url, str(dest_path)]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            return CloneResult(
                status="clone_failed", error_message="clone timed out"
            )
        except FileNotFoundError:
            return CloneResult(
                status="clone_failed", error_message="git binary not found"
            )

        stderr_text = stderr.decode() if stderr else ""
        if proc.returncode != 0:
            lower = stderr_text.lower()
            if "authentication failed" in lower or "access denied" in lower:
                return CloneResult(
                    status="authentication_failed", error_message=stderr_text
                )
            return CloneResult(
                status="clone_failed", error_message=stderr_text
            )

        if (
            disk_limit_bytes > 0
            and _dir_size(str(dest_path)) > disk_limit_bytes
        ):
            _rmtree(str(dest_path))
            return CloneResult(status="repo_too_large")

        return CloneResult(status="success", repo_path=str(dest_path))
    finally:
        if script_path is not None:
            try:
                os.unlink(script_path)
            except OSError:
                pass


def _dir_size(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def _rmtree(path: str) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
