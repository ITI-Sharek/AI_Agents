"""Semantic Matching database wiring (Phase 8).

The matching index is an independent PostgreSQL + pgvector database,
configured through ``SEMANTIC_MATCHING_DATABASE_URL``. The storage layer
(``storage.py``) is intentionally driver-agnostic: it consumes an injected
"asyncpg-style" connection (``execute`` / ``fetchrow`` / ``fetch``). This
module supplies the real asyncpg-backed wiring that makes the store work
against PostgreSQL + pgvector.

The vector codec is required because pgvector is not a native asyncpg type:

- encoding: pgvector accepts only ``[1, 2, 3]`` literals, but asyncpg would
  otherwise encode Python lists as ``{1,2,3}`` array literals, which
  pgvector rejects;
- decoding: without a codec asyncpg returns stored vectors as raw text.

A ``vector`` type codec is registered on every pooled connection so the
``$n::vector`` casts used by the store round-trip Python ``list[float]``.

asyncpg is imported lazily so the feature keeps the store's driver-agnostic
design: importing this module never requires the driver, and a clear error
is raised only when a connection is actually requested without it installed.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable


# ── Vector codec ──────────────────────────────────────────────────────────────


def _vector_literal(vector: list[float]) -> str:
    """Encode a Python vector as a pgvector ``[1, 2, 3]`` literal."""
    return "[" + ",".join(repr(value) for value in vector) + "]"


def _parse_vector(value: Any) -> list[float]:
    """Decode a pgvector literal (e.g. ``[1,2,3]``) into a list of floats."""
    if isinstance(value, list):
        return [float(item) for item in value]
    content = str(value).strip()
    if content.startswith("[") and content.endswith("]"):
        content = content[1:-1]
    if not content:
        return []
    return [float(piece) for piece in content.split(",")]


# ── Connection wiring ─────────────────────────────────────────────────────────


class _AsyncpgVectorAdapter:
    """Minimal asyncpg-style connection duck type.

    Wraps an ``asyncpg.Pool`` so the driver-agnostic
    ``PostgresSemanticMatchingStore`` (which calls ``execute`` /
    ``fetchrow`` / ``fetch``) works against it.
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def execute(self, query: str, *args: Any) -> str:
        return await self._pool.execute(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Any:
        return await self._pool.fetchrow(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        return await self._pool.fetch(query, *args)


def create_connection_provider(
    database_url: str,
) -> Callable[[], Awaitable[Any]]:
    """Build the ``connection_provider`` injected into the matching store.

    A single asyncpg pool is created lazily on first use and reused (the
    store asks for a connection per operation). A ``vector`` type codec is
    registered on every pooled connection so pgvector values round-trip as
    Python ``list[float]``, the format the store's ``$n::vector`` casts and
    its embedding decoding expect.

    The pool has no explicit close hook; its lifecycle is owned by the
    deployment wiring (the store protocol has no shutdown contract).
    """

    state: dict[str, Any] = {"pool": None}

    async def _init_connection(connection: Any) -> None:
        await connection.set_type_codec(
            "vector",
            encoder=_vector_literal,
            decoder=_parse_vector,
            format="text",
        )

    async def _pool() -> Any:
        if state["pool"] is None:
            try:
                import asyncpg  # noqa: PLC0415 (lazy: driver-agnostic design)
            except ImportError as exc:  # pragma: no cover - environment check
                raise RuntimeError(
                    "asyncpg is not installed; it is required to connect to "
                    "the Semantic Matching PostgreSQL + pgvector index. "
                    "Install asyncpg in the deployment environment."
                ) from exc
            state["pool"] = await asyncpg.create_pool(
                database_url, init=_init_connection
            )
        return state["pool"]

    async def provider() -> _AsyncpgVectorAdapter:
        return _AsyncpgVectorAdapter(await _pool())

    return provider