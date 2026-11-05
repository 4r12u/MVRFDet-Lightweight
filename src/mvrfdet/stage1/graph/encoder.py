from __future__ import annotations

import networkx as nx
import numpy as np

from mvrfdet.core.types import APICallGraph
from .api_graph_features import APIGraphStatisticsEncoder
from .cfg_features import CFGStatisticsEncoder
from .fusion import ConcatenationGraphFusion


class StatisticalGraphEncoder:
    """API-CG statistics + CFG statistics + concatenation -> Vg[1024]."""

    def __init__(self):
        self.api_encoder = APIGraphStatisticsEncoder(512)
        self.cfg_encoder = CFGStatisticsEncoder(512)
        self.fusion = ConcatenationGraphFusion()

    def encode(self, api_call_graph: APICallGraph, cfgs: list[nx.DiGraph], sensitive_apis: set[str] | None = None) -> np.ndarray:
        gcg = self.api_encoder.encode(api_call_graph, sensitive_apis=sensitive_apis)
        gcfg = self.cfg_encoder.encode(cfgs)
        vg = self.fusion.fuse(gcg, gcfg)
        if vg.shape != (1024,):
            raise RuntimeError(f"graph encoder produced {vg.shape}")
        return vg
