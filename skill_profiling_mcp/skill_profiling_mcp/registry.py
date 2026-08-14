"""In-memory registry of acquired Docker sandboxes.

Phase 5: ``acquire_repository`` registers a lease describing the acquired
sandbox (repository reference only, never credentials). Later tools, such
as ``filter_contributor_code``, validate their ``sandbox_identifier``
against this registry and fail closed when it is unknown.

Phase 6: ``filter_contributor_code`` attaches a ``ContributorScope`` to
the lease. The analysis tools (``analyze_static`` / ``analyze_graph``)
only ever analyze this registered contributor scope and fail closed when
no scope is registered for the sandbox.

Phase 24: every lease is TTL-bounded (``SANDBOX_TTL_SECONDS``): an
expired lease is treated as unknown by ``get`` (fail closed), is pruned
by the sandbox sweep, and its container is removed on the next
acquisition. The registry is guarded by a reentrant lock because it is
mutated from asyncio ``to_thread`` workers and the event loop, and
``RepositoryAcquirer.acquire`` holds that lock across
sweep → container start → lease registration so a concurrent request can
never sweep a just-started, not-yet-registered container.
"""

import os
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass

from skill_profiling_mcp.sandbox import SANDBOX_IMAGE, SandboxError

# Bounded sandbox lifetime: a lease (and its container) live at most
# SANDBOX_TTL_SECONDS after registration. The orphan sweep on the next
# acquisition reaps the container once the lease has expired.
DEFAULT_SANDBOX_TTL_SECONDS = 3600
SANDBOX_TTL_ENV = "SKILL_PROFILING_SANDBOX_TTL_SECONDS"


def _sandbox_ttl_seconds() -> int:
    """Return the sandbox TTL in seconds, read from the environment.

    This package has no other configuration mechanism, so a single
    environment variable (``SANDBOX_TTL_ENV``) is the smallest possible
    override point. Missing, non-numeric, or non-positive values fall
    back to ``DEFAULT_SANDBOX_TTL_SECONDS`` (fail safe).
    """
    raw = os.environ.get(SANDBOX_TTL_ENV)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    return DEFAULT_SANDBOX_TTL_SECONDS


SANDBOX_TTL_SECONDS = _sandbox_ttl_seconds()


@dataclass(frozen=True)
class ContributorScope:
    """Deterministic record of a ``filter_contributor_code`` result.

    Contains the contributor identifier, the scoped workspace path, and
    the manifest of owned paths (paths only — never file contents).
    """

    contributor_identifier: str
    workspace_path: str
    manifest: tuple[str, ...]


@dataclass(frozen=True)
class SandboxLease:
    """Safe, credential-free metadata for an acquired repository sandbox.

    ``expires_at`` is a monotonic-clock deadline assigned at construction
    (``SANDBOX_TTL_SECONDS`` after creation): an expired lease is treated
    as unknown by the registry (fail closed) and its container is swept
    on the next acquisition, bounding every sandbox's lifetime.
    """

    sandbox_identifier: str
    repository_identifier: str
    repo_url: str
    image: str = SANDBOX_IMAGE
    scope: ContributorScope | None = None
    expires_at: float | None = None

    def __post_init__(self) -> None:
        if self.expires_at is None:
            object.__setattr__(self, "expires_at", time.monotonic() + SANDBOX_TTL_SECONDS)

    def is_expired(self, now: float | None = None) -> bool:
        """Return True when the lease TTL has elapsed (fail closed)."""
        if self.expires_at is None:
            return False
        return (now if now is not None else time.monotonic()) >= self.expires_at


class SandboxRegistry:
    """Maps ``sandbox_identifier`` to its credential-free lease.

    All operations are guarded by a reentrant lock: the registry is
    mutated from asyncio ``to_thread`` workers (the acquirer's docker
    calls, the filter, the analyzers) while the event loop may touch it
    too, and ``RepositoryAcquirer.acquire`` holds the lock across the
    orphan sweep, container start, and lease registration so that
    registration is atomic with respect to sweeping.
    """

    def __init__(self) -> None:
        self._leases: dict[str, SandboxLease] = {}
        self._lock = threading.RLock()

    def locked(self) -> AbstractContextManager[None]:
        """Return a context manager guarding all registry operations."""
        return self._lock

    def register(self, lease: SandboxLease) -> None:
        """Record a lease keyed by its sandbox identifier."""
        with self._lock:
            self._leases[lease.sandbox_identifier] = lease

    def get(self, sandbox_identifier: str) -> SandboxLease | None:
        """Return the live lease for a sandbox identifier, or None.

        Returns None for unknown sandbox identifiers AND for leases whose
        TTL has elapsed — an expired sandbox is never usable (fail
        closed; the sweep reaps its container).
        """
        with self._lock:
            lease = self._leases.get(sandbox_identifier)
            if lease is not None and lease.is_expired():
                return None
            return lease

    def sandbox_ids(self) -> tuple[str, ...]:
        """Return the registered sandbox identifiers (for orphan sweeps)."""
        with self._lock:
            return tuple(sorted(self._leases))

    def expired_ids(self) -> tuple[str, ...]:
        """Return identifiers of leases whose TTL has elapsed."""
        now = time.monotonic()
        with self._lock:
            return tuple(
                sorted(sid for sid, lease in self._leases.items() if lease.is_expired(now))
            )

    def attach_scope(self, sandbox_identifier: str, scope: ContributorScope) -> None:
        """Attach a contributor scope to an existing lease.

        Replaces the lease with an identical one carrying the scope
        (frozen dataclass, so the mapping value is recreated); the
        original TTL deadline is preserved, never reset. Raises
        ``SandboxError`` when the sandbox identifier is unknown — scope
        registration fails closed.
        """
        with self._lock:
            lease = self._leases.get(sandbox_identifier)
            if lease is None:
                raise SandboxError("sandbox unavailable: unknown sandbox_identifier")
            self._leases[sandbox_identifier] = SandboxLease(
                sandbox_identifier=lease.sandbox_identifier,
                repository_identifier=lease.repository_identifier,
                repo_url=lease.repo_url,
                image=lease.image,
                scope=scope,
                expires_at=lease.expires_at,
            )

    def remove(self, sandbox_identifier: str) -> None:
        """Drop a lease (idempotent)."""
        with self._lock:
            self._leases.pop(sandbox_identifier, None)

    def clear(self) -> None:
        """Drop all leases (test hygiene)."""
        with self._lock:
            self._leases.clear()


sandbox_registry = SandboxRegistry()

__all__ = [
    "DEFAULT_SANDBOX_TTL_SECONDS",
    "SANDBOX_TTL_ENV",
    "SANDBOX_TTL_SECONDS",
    "ContributorScope",
    "SandboxLease",
    "SandboxRegistry",
    "sandbox_registry",
]
