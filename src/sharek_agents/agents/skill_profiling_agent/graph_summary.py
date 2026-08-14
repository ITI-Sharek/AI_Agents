"""Deterministic compact summarization of stored Graphify output (Phase 28).

Large Graphify outputs are never dumped into the LLM context window
unbounded. The Context Store keeps the ORIGINAL graph output as-is and,
only when the output exceeds a configurable character threshold, stores
an ADDITIONAL compact summary under the ``*_graph_summary`` key. The
summary is an extra representation — never a replacement — and the
original graph is never overwritten or deleted.

The decision is fully deterministic: the size is the character length
of the stored graph output, and the threshold is compared directly. The
LLM never decides whether a graph is large, and no Redis, database,
RAG, embeddings, or retrieval is involved.

The summary is extractive, not generative: it preserves the metric and
evidence fields already present in the Graphify output (node/file and
edge/relation counts, relation counts by type, inheritance depth,
coupling, circular-import count, analyzer status, and messages) and
invents no information. No LLM summarization step is used in this
phase; the existing configured LLM infrastructure remains untouched.
"""

from __future__ import annotations

import json
from typing import Any

# Safe default: graph outputs up to this many characters are kept in
# full and are NOT summarized; larger outputs get a compact summary.
DEFAULT_GRAPH_SUMMARY_THRESHOLD_CHARS = 4000

# Metric and evidence fields preserved from a Graphify payload. Only
# fields actually present in the output are copied — nothing is
# invented. ``relations`` (a dict of relation-type -> count) is handled
# separately below.
_GRAPH_METRIC_FIELDS = (
    "status",
    "analyzer",
    "graph_available",
    "node_count",
    "edge_count",
    "relation_count",
    "relation_truncated",
    "inheritance_depth",
    "coupling",
    "circular_import_count",
    "error_message",
    "message",
)


def is_graph_too_large(graph_output: str, threshold_chars: int) -> bool:
    """Deterministic size check: whether the graph output is too large.

    The size of a graph output is the number of characters in its
    stored JSON text. An output strictly larger than the threshold is
    too large and must be summarized; anything at or below the
    threshold is kept in full without a summary.
    """
    return len(graph_output) > threshold_chars


def summarize_graph_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Reduce a Graphify evidence payload to its compact summary.

    Preserves only information already present in the payload: the
    metric/evidence fields the Graphify output carries (node, edge, and
    relation counts, inheritance depth, coupling, circular-import
    count, analyzer status, messages) and the per-relation-type counts
    (``relations`` is a ``dict`` of relation type -> count in the
    Graphify output). Nothing is invented, derived, or re-analyzed.
    """
    summary: dict[str, Any] = {}
    for name in _GRAPH_METRIC_FIELDS:
        if name in payload:
            summary[name] = payload[name]
    relations = payload.get("relations")
    if isinstance(relations, dict):
        summary["relations"] = dict(relations)
    return summary


def build_graph_summary(graph_output: str) -> str | None:
    """Return the compact JSON summary of a stored graph output.

    Returns ``None`` when the output is not a JSON object — such an
    output can never be summarized and is only ever stored as-is.
    """
    try:
        payload = json.loads(graph_output)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    return json.dumps(summarize_graph_payload(payload), ensure_ascii=False)


__all__ = [
    "DEFAULT_GRAPH_SUMMARY_THRESHOLD_CHARS",
    "build_graph_summary",
    "is_graph_too_large",
    "summarize_graph_payload",
]