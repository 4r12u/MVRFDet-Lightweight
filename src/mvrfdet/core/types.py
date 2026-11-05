from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .constants import METADATA_DIM, CODE_DIM, GRAPH_DIM


@dataclass(frozen=True)
class APINode:
    id: str
    api_name: str
    file: str
    line: int
    column: int
    function: str = "<module>"
    kind: str = "api"


@dataclass
class APICallGraph:
    nodes: dict[str, APINode] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    backend: str = "unknown"

    def add_node(self, node: APINode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, source: str, target: str) -> None:
        if source in self.nodes and target in self.nodes:
            self.edges.append((source, target))

    def to_networkx(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for node_id, node in self.nodes.items():
            graph.add_node(
                node_id,
                api_name=node.api_name,
                file=node.file,
                line=node.line,
                column=node.column,
                function=node.function,
                kind=node.kind,
            )
        graph.add_edges_from(self.edges)
        return graph

    @classmethod
    def from_networkx(cls, graph: nx.DiGraph, backend: str = "networkx") -> "APICallGraph":
        result = cls(backend=backend)
        for node_id, attrs in graph.nodes(data=True):
            sid = str(node_id)
            result.add_node(
                APINode(
                    id=sid,
                    api_name=str(attrs.get("api_name", sid)),
                    file=str(attrs.get("file", "")),
                    line=int(attrs.get("line", 0) or 0),
                    column=int(attrs.get("column", 0) or 0),
                    function=str(attrs.get("function", "<module>")),
                    kind=str(attrs.get("kind", "api")),
                )
            )
        for a, b in graph.edges():
            result.add_edge(str(a), str(b))
        return result


@dataclass(frozen=True)
class SuspiciousAPIOccurrence:
    node_id: str
    api_name: str
    file: Path
    line: int
    column: int
    function: str = "<module>"


@dataclass(frozen=True)
class CodeContextSnippet:
    api_name: str
    source_file: Path
    center_line: int
    text: str
    token_count: int
    digest: str


@dataclass
class MultiViewFeatures:
    metadata: np.ndarray
    code: np.ndarray
    graph: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.metadata.shape != (METADATA_DIM,):
            raise ValueError(f"metadata feature must be ({METADATA_DIM},), got {self.metadata.shape}")
        if self.code.shape != (CODE_DIM,):
            raise ValueError(f"code feature must be ({CODE_DIM},), got {self.code.shape}")
        if self.graph.shape != (GRAPH_DIM,):
            raise ValueError(f"graph feature must be ({GRAPH_DIM},), got {self.graph.shape}")
        for name, vector in (("metadata", self.metadata), ("code", self.code), ("graph", self.graph)):
            if not np.isfinite(vector).all():
                raise ValueError(f"{name} feature contains NaN/Inf")

    def save_npz(self, path: str | Path) -> None:
        self.validate()
        np.savez_compressed(path, vm=self.metadata, vc=self.code, vg=self.graph)

    @classmethod
    def load_npz(cls, path: str | Path) -> "MultiViewFeatures":
        with np.load(path, allow_pickle=False) as data:
            result = cls(
                metadata=np.asarray(data["vm"], dtype=np.float32),
                code=np.asarray(data["vc"], dtype=np.float32),
                graph=np.asarray(data["vg"], dtype=np.float32),
            )
        result.validate()
        return result
