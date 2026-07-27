from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


CLONE_NEWNET = 0x40000000

_NAMESPACE_AVAILABLE: Optional[bool] = None


def _check_namespace_available() -> bool:
    global _NAMESPACE_AVAILABLE
    if _NAMESPACE_AVAILABLE is not None:
        return _NAMESPACE_AVAILABLE
    try:
        os.unshare(CLONE_NEWNET)
        _NAMESPACE_AVAILABLE = True
        return True
    except (PermissionError, OSError, AttributeError):
        _NAMESPACE_AVAILABLE = False
        return False


class SandboxResources:
    def __init__(
        self,
        work_dir: Path,
        disk_limit_bytes: int,
        cpu_time_limit: int,
        memory_limit_bytes: int,
    ) -> None:
        self.work_dir = work_dir
        self._disk_limit = disk_limit_bytes
        self._cpu_limit = cpu_time_limit
        self._mem_limit = memory_limit_bytes

    def disk_usage_bytes(self) -> int:
        total = 0
        for dirpath, _dirnames, filenames in os.walk(self.work_dir):
            for f in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
        return total

    def disk_usage_mb(self) -> float:
        return self.disk_usage_bytes() / (1024 * 1024)

    def within_disk_limit(self) -> bool:
        return self.disk_usage_bytes() <= self._disk_limit

    def _apply_limits(self) -> None:
        try:
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (self._cpu_limit, self._cpu_limit + 10),
            )
        except (ValueError, resource.error):
            pass
        try:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self._mem_limit, self._mem_limit),
            )
        except (ValueError, resource.error):
            pass
        try:
            resource.setrlimit(
                resource.RLIMIT_FSIZE,
                (100 * 1024 * 1024, 100 * 1024 * 1024),
            )
        except (ValueError, resource.error):
            pass
        if _check_namespace_available():
            try:
                os.unshare(CLONE_NEWNET)
            except OSError:
                pass

    def run(
        self,
        cmd: list[str],
        timeout: Optional[float] = None,
        **kwargs: object,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            preexec_fn=self._apply_limits,
            timeout=timeout,
            **kwargs,
        )

    async def run_async(
        self,
        cmd: list[str],
        timeout: Optional[float] = None,
        **kwargs: object,
    ):
        import asyncio

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            preexec_fn=self._apply_limits,
            **kwargs,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            return proc.returncode or 0, stdout, stderr
        except asyncio.TimeoutError:
            try:
                proc.kill()
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError, PermissionError):
                pass
            raise


@contextmanager
def sandbox_context(
    disk_limit_mb: int = 500,
    cpu_time_limit: int = 120,
    memory_limit_mb: int = 1024,
) -> Iterator[SandboxResources]:
    tmp = tempfile.mkdtemp(prefix="code-analysis-sandbox-")
    resources = SandboxResources(
        work_dir=Path(tmp),
        disk_limit_bytes=disk_limit_mb * 1024 * 1024,
        cpu_time_limit=cpu_time_limit,
        memory_limit_bytes=memory_limit_mb * 1024 * 1024,
    )
    try:
        yield resources
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
