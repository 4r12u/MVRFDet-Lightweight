from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from mvrfdet.core.types import APICallGraph, CodeContextSnippet, MultiViewFeatures, SuspiciousAPIOccurrence
from mvrfdet.static_analysis.cfg import build_cfg
from mvrfdet.static_analysis.scalpel_adapter import StaticAnalysisAdapter
from mvrfdet.stage1.code.context import extract_contexts
from mvrfdet.stage1.code.encoder import StaticCodeEncoder
from mvrfdet.stage1.graph.encoder import StatisticalGraphEncoder
from mvrfdet.stage1.metadata.encoder import StaticMetadataEncoder
from mvrfdet.stage1.metadata.parser import locate_pkg_info, parse_pkg_info
from mvrfdet.stage1.suspicious_api.locator import locate_suspicious_apis


@dataclass
class PipelineArtifacts:
    """Intermediate results retained for per-package inspection."""

    features: MultiViewFeatures
    metadata: dict[str, str]
    api_graph: APICallGraph
    suspicious: list[SuspiciousAPIOccurrence]
    contexts: list[CodeContextSnippet]
    cfgs: list[nx.DiGraph]


class LightweightFeaturePipeline:
    """End-to-end feature extraction for Vm[768], Vc[768], Vg[1024]."""

    def __init__(self, static_backend: str = "auto", max_context_tokens: int = 4096):
        self.analyzer = StaticAnalysisAdapter(static_backend)
        self.metadata_encoder = StaticMetadataEncoder()
        self.code_encoder = StaticCodeEncoder()
        self.graph_encoder = StatisticalGraphEncoder()
        self.max_context_tokens = max_context_tokens

    def extract(self, package_path: str | Path, sensitive_apis: set[str]) -> MultiViewFeatures:
        return self.extract_with_artifacts(package_path, sensitive_apis).features

    def extract_with_artifacts(self, package_path: str | Path, sensitive_apis: set[str]) -> PipelineArtifacts:
        package_path = Path(package_path)
        pkg_info_path = locate_pkg_info(package_path)
        metadata = parse_pkg_info(pkg_info_path)
        vm = self.metadata_encoder.encode(metadata)

        api_graph = self.analyzer.extract_api_call_graph(package_path)
        suspicious = locate_suspicious_apis(api_graph, sensitive_apis)
        contexts = extract_contexts(suspicious, max_tokens=self.max_context_tokens, deduplicate_identical=True)
        vc = self.code_encoder.encode(contexts)

        cfgs = [build_cfg(snippet.text) for snippet in contexts]
        vg = self.graph_encoder.encode(api_graph, cfgs, sensitive_apis=sensitive_apis)

        features = MultiViewFeatures(
            metadata=vm,
            code=vc,
            graph=vg,
            diagnostics={
                "package": str(package_path),
                "pkg_info": str(pkg_info_path),
                "static_backend": api_graph.backend,
                "api_nodes": len(api_graph.nodes),
                "api_edges": len(api_graph.edges),
                "suspicious_occurrences": len(suspicious),
                "unique_contexts": len(contexts),
                "cfg_count": len(cfgs),
                "suspicious_api_names": sorted({item.api_name for item in suspicious}),
            },
        )
        features.validate()
        return PipelineArtifacts(features, metadata, api_graph, suspicious, contexts, cfgs)
