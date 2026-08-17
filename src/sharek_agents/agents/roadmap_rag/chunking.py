"""Roadmap chunking: split roadmap text into meaningful, ordered chunks.

Roadmap text generally contains ordered steps such as:

    1. Learn HTTP fundamentals
    2. Build REST APIs
    3. Authentication
    ...

The chunker keeps the step/order information: chunks are built by grouping
consecutive non-empty lines (each line is one step/item), and the returned
list order IS the ``chunk_index`` order. A line longer than the chunk size
becomes a chunk of its own. Blank text produces no chunks.
"""

from __future__ import annotations

from sharek_agents.config import settings


def chunk_roadmap(
    roadmap_text: str,
    *,
    chunk_size: int | None = None,
) -> list[str]:
    """Split roadmap text into ordered chunks.

    Args:
        roadmap_text: The roadmap knowledge text (ordered steps).
        chunk_size: Approximate character budget per chunk
            (defaults to ``settings.chunk_size``).

    Returns:
        The chunk contents in order; each list position is the chunk's
        ``chunk_index``. Empty for blank/whitespace-only input.
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    lines = [line.strip() for line in (roadmap_text or "").splitlines()]
    steps = [line for line in lines if line]
    if not steps:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for step in steps:
        if current and current_length + len(step) + 1 > chunk_size:
            chunks.append("\n".join(current))
            current = []
            current_length = 0
        current.append(step)
        current_length += len(step) + 1
        if current_length > chunk_size:
            chunks.append("\n".join(current))
            current = []
            current_length = 0
    if current:
        chunks.append("\n".join(current))
    return chunks
