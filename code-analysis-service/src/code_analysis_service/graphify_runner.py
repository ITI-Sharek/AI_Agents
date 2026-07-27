from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from .models import (
    CircularImport,
    GraphEdge,
    GraphNode,
    GraphRelationsEvidence,
)

logger = logging.getLogger(__name__)

_FRAMEWORK_KEYWORDS: set[str] = {
    "django", "flask", "fastapi", "spring", "react", "angular", "vue",
    "express", "rails", "laravel", "aspnet", "entityframework", "hibernate",
    "sqlalchemy", "tensorflow", "pytorch", "jquery", "bootstrap",
    "tailwind", "nextjs", "nuxt", "gatsby", "symfony", "codeigniter",
    "cake", "yii", "zend", "struts", "javamail", "log4j", "junit",
    "numpy", "pandas", "requests", "asyncio", "aiohttp",
}


def _strip_framework_names(text: str) -> str:
    tokens = text.split()
    filtered: list[str] = []
    for token in tokens:
        clean = token.strip(".,;:!?\"'()[]{}").lower()
        if clean in _FRAMEWORK_KEYWORDS:
            filtered.append("[module]")
        else:
            filtered.append(token)
    return " ".join(filtered)


def _build_coupling_summary(
    coupling: Optional[float],
    inheritance_depth: Optional[int],
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    circular_imports: list[CircularImport],
) -> str:
    parts: list[str] = []
    if coupling is not None:
        parts.append(f"coupling={coupling:.3f}")
    if inheritance_depth is not None:
        parts.append(f"max_inheritance_depth={inheritance_depth}")
    parts.append(f"nodes={len(nodes)}")
    parts.append(f"edges={len(edges)}")
    ci_count = len(circular_imports)
    if ci_count > 0:
        ci_nodes = sum(len(c.chain) for c in circular_imports)
        parts.append(f"circular_import_chains={ci_count}")
        parts.append(f"circular_import_nodes={ci_nodes}")
    raw = "; ".join(parts)
    raw = _strip_framework_names(raw)
    if len(raw) > 500:
        raw = raw[:497] + "..."
    return raw


def _classify_node_type(
    node_data: dict[str, Any],
) -> tuple[str, bool]:
    raw_type = node_data.get("type") or node_data.get("node_type")
    file_type = node_data.get("file_type", "")
    label: str = node_data.get("label", node_data.get("id", ""))

    if raw_type:
        lower = raw_type.lower()
        if lower in ("module", "class", "function", "method", "file", "package"):
            return lower, False
        return raw_type, True

    if label.endswith("()"):
        return "function", False
    if file_type == "code" and label and label[0].isupper():
        return "heuristic:class", True
    if file_type == "code":
        return "heuristic:code_element", True
    return "heuristic:unknown", True


def _extract_edges(data: dict) -> list[dict]:
    if "edges" in data and isinstance(data["edges"], list):
        return data["edges"]
    if "links" in data and isinstance(data["links"], list):
        return data["links"]
    return []


async def run_graphify(
    cloned_repo_path: str, timeout_seconds: int = 60
) -> GraphRelationsEvidence:
    graphify_bin = shutil.which("graphify")
    if graphify_bin is None:
        return GraphRelationsEvidence(status="tool_unavailable")

    try:
        proc = await asyncio.create_subprocess_exec(
            graphify_bin,
            "extract",
            cloned_repo_path,
            "--code-only",
            "--no-cluster",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        return GraphRelationsEvidence(status="timeout")
    except FileNotFoundError:
        return GraphRelationsEvidence(status="tool_unavailable")
    except Exception as exc:
        return GraphRelationsEvidence(
            status="error", error_message=str(exc)
        )

    if proc.returncode != 0:
        return GraphRelationsEvidence(
            status="error",
            error_message=(stderr or b"").decode()[:500],
        )

    graph_file = Path(cloned_repo_path) / "graphify-out" / "graph.json"
    if not graph_file.exists():
        return GraphRelationsEvidence(
            status="error",
            error_message="graphify did not produce graph.json",
        )

    try:
        data = json.loads(graph_file.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return GraphRelationsEvidence(
            status="error", error_message=f"cannot read graph: {exc}"
        )

    raw_nodes = data.get("nodes", [])
    nodes: list[GraphNode] = []
    for n in raw_nodes:
        node_id = n.get("id", "")
        label = n.get("label", node_id)
        file_path: str = n.get("source_file", "")
        resolved_type, is_heuristic = _classify_node_type(n)
        nodes.append(
            GraphNode(
                id=label or node_id,
                node_type=resolved_type,
                file_path=file_path,
                heuristic_label=is_heuristic,
            )
        )

    raw_edges = _extract_edges(data)
    edges: list[GraphEdge] = []
    for e in raw_edges:
        edges.append(
            GraphEdge(
                source=e.get("source", ""),
                target=e.get("target", ""),
                relation=e.get("relation", ""),
            )
        )

    coupling = _compute_coupling(nodes, edges)
    inheritance_depth = _compute_inheritance_depth(edges)
    circular_imports = _detect_circular_imports(edges)

    coupling_summary = _build_coupling_summary(
        coupling, inheritance_depth, nodes, edges, circular_imports
    )

    return GraphRelationsEvidence(
        status="success",
        nodes=nodes,
        edges=edges,
        inheritance_depth=inheritance_depth,
        coupling=coupling,
        coupling_summary=coupling_summary,
        circular_imports=circular_imports,
    )


def _compute_coupling(
    nodes: list[GraphNode], edges: list[GraphEdge]
) -> Optional[float]:
    if not nodes:
        return None
    return round(len(edges) / len(nodes), 4)


def _compute_inheritance_depth(edges: list[GraphEdge]) -> Optional[int]:
    parents: dict[str, list[str]] = {}
    children: set[str] = set()
    for e in edges:
        if e.relation in ("inherits", "extends", "implements"):
            child = e.source
            parent = e.target
            parents.setdefault(child, []).append(parent)
            children.add(child)
    if not parents:
        return None
    max_depth = 0
    memo: dict[str, int] = {}

    def _depth(node: str, visited: set[str]) -> int:
        if node in memo:
            return memo[node]
        if node in visited:
            return 0
        visited.add(node)
        max_p = 0
        for p in parents.get(node, []):
            max_p = max(max_p, 1 + _depth(p, visited))
        visited.discard(node)
        memo[node] = max_p
        return max_p

    for c in children:
        max_depth = max(max_depth, _depth(c, set()))
    return max_depth if max_depth > 0 else None


def _detect_circular_imports(
    edges: list[GraphEdge],
) -> list[CircularImport]:
    graph: dict[str, list[str]] = {}
    for e in edges:
        if e.relation in ("imports", "import"):
            graph.setdefault(e.source, []).append(e.target)

    cycles: list[CircularImport] = []
    visited: set[str] = set()
    rec_stack: list[str] = []

    def _dfs(node: str) -> None:
        visited.add(node)
        rec_stack.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                _dfs(neighbor)
            elif neighbor in rec_stack:
                cycle_start = rec_stack.index(neighbor)
                chain = rec_stack[cycle_start:] + [neighbor]
                cycles.append(CircularImport(chain=list(chain)))
        rec_stack.pop()

    for node in list(graph.keys()):
        if node not in visited:
            _dfs(node)

    return cycles
