from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnalysisIssue:
    line: int = 0
    column: int = 0
    severity: str = "info"
    message: str = ""
    rule_id: str = ""
    file_path: str = ""
    node_type: Optional[str] = None
    heuristic_label: bool = False


@dataclass
class InheritanceRelation:
    child_class: str
    parent_class: str
    file_path: str = ""
    line: int = 0


@dataclass
class CircularImport:
    chain: list[str] = field(default_factory=list)
    file_paths: list[str] = field(default_factory=list)


@dataclass
class StructuralGraph:
    inheritance_relationships: list[InheritanceRelation] = field(default_factory=list)
    coupling: Optional[float] = None
    circular_imports: list[CircularImport] = field(default_factory=list)


@dataclass
class StaticAnalysisEvidence:
    status: str = "success"
    language: str = ""
    files_analyzed: int = 0
    complexity: Optional[float] = None
    maintainability_index: Optional[float] = None
    issues: list[AnalysisIssue] = field(default_factory=list)
    structure: StructuralGraph = field(default_factory=StructuralGraph)
    error_message: Optional[str] = None


@dataclass
class CloneResult:
    status: str = "success"
    repo_path: str = ""
    error_message: Optional[str] = None


@dataclass
class GraphNode:
    id: str = ""
    node_type: str = ""
    file_path: str = ""
    heuristic_label: bool = False


@dataclass
class GraphEdge:
    source: str = ""
    target: str = ""
    relation: str = ""


@dataclass
class GraphRelationsEvidence:
    status: str = "success"
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    inheritance_depth: Optional[int] = None
    coupling: Optional[float] = None
    coupling_summary: str = ""
    circular_imports: list[CircularImport] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class AnalysisResult:
    static_analysis: Optional[StaticAnalysisEvidence] = None
    graph_relations: Optional[GraphRelationsEvidence] = None


EXTENSIONS: dict[str, list[str]] = {
    "python": [".py"],
    "javascript": [".js", ".jsx", ".mjs"],
    "typescript": [".ts", ".tsx"],
    "java": [".java"],
    "go": [".go"],
    "rust": [".rs"],
    "ruby": [".rb"],
    "php": [".php"],
    "csharp": [".cs"],
}
