from __future__ import annotations

import numpy as np


class ConcatenationGraphFusion:
    """Lightweight replacement for the paper's Selection Feature Fusion."""

    def fuse(self, api_graph_vector: np.ndarray, cfg_vector: np.ndarray) -> np.ndarray:
        if api_graph_vector.shape != (512,) or cfg_vector.shape != (512,):
            raise ValueError("Expected Gcg[512] and Gcfg[512]")
        return np.concatenate([api_graph_vector, cfg_vector]).astype(np.float32)
