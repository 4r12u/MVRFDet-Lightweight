from __future__ import annotations

import numpy as np
from sklearn.feature_extraction import FeatureHasher

from mvrfdet.core.types import APICallGraph
from mvrfdet.graph_utils import graph_statistics


class APIGraphStatisticsEncoder:
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.hasher = FeatureHasher(n_features=dimension, input_type="dict", alternate_sign=False)

    def encode(self, graph: APICallGraph, sensitive_apis: set[str] | None = None) -> np.ndarray:
        nx_graph = graph.to_networkx()
        features = graph_statistics(nx_graph, sensitive_apis=sensitive_apis, prefix="api_cg")
        return self.hasher.transform([features]).toarray()[0].astype(np.float32)
