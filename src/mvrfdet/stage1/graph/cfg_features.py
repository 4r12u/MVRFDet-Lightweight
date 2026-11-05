from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.feature_extraction import FeatureHasher

from mvrfdet.graph_utils import graph_statistics


def merge_cfgs(cfgs: list[nx.DiGraph]) -> nx.DiGraph:
    if not cfgs:
        return nx.DiGraph()
    merged = nx.DiGraph()
    for index, cfg in enumerate(cfgs):
        mapping = {node: f"cfg{index}:{node}" for node in cfg.nodes}
        renamed = nx.relabel_nodes(cfg, mapping, copy=True)
        merged = nx.compose(merged, renamed)
    return merged


class CFGStatisticsEncoder:
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.hasher = FeatureHasher(n_features=dimension, input_type="dict", alternate_sign=False)

    def encode(self, cfgs: list[nx.DiGraph]) -> np.ndarray:
        graph = merge_cfgs(cfgs)
        if graph.number_of_nodes() == 0:
            return np.zeros(self.dimension, dtype=np.float32)
        features = graph_statistics(graph, sensitive_apis=None, prefix="cfg")
        # Add control-flow-specific counts.
        out_degrees = dict(graph.out_degree())
        in_degrees = dict(graph.in_degree())
        features["cfg:branch_nodes"] = float(sum(1 for degree in out_degrees.values() if degree > 1))
        features["cfg:merge_nodes"] = float(sum(1 for degree in in_degrees.values() if degree > 1))
        features["cfg:entry_nodes"] = float(sum(1 for _, attrs in graph.nodes(data=True) if attrs.get("entry")))
        features["cfg:exit_nodes"] = float(sum(1 for _, attrs in graph.nodes(data=True) if attrs.get("exit") or attrs.get("terminal")))
        # E-N+2P, generalized for disconnected undirected components.
        components = nx.number_connected_components(graph.to_undirected()) if graph.number_of_nodes() else 0
        features["cfg:cyclomatic_complexity"] = float(graph.number_of_edges() - graph.number_of_nodes() + 2 * components)
        return self.hasher.transform([features]).toarray()[0].astype(np.float32)
