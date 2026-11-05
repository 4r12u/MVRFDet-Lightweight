from __future__ import annotations

import numpy as np
import torch

from mvrfdet.core.constants import ALIGNED_DIM, MATRIX_SHAPE, METADATA_DIM, CODE_DIM, GRAPH_DIM
from mvrfdet.core.types import MultiViewFeatures


def _pad(vector: np.ndarray, target: int = ALIGNED_DIM) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    if vector.size > target:
        raise ValueError(f"Cannot pad vector of length {vector.size} to smaller target {target}")
    if vector.size == target:
        return vector.copy()
    return np.pad(vector, (0, target - vector.size), mode="constant", constant_values=0).astype(np.float32)


def _l2_normalize(vector: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vector, ord=2))
    return (vector / (norm + eps)).astype(np.float32) if norm > 0 else vector.astype(np.float32)


class FeatureMatrixGenerator:
    """Paper-aligned Stage 2: pad -> per-view L2 norm -> reshape -> stack."""

    def __init__(self, eps: float = 1.0e-12):
        self.eps = eps

    def generate_numpy(self, features: MultiViewFeatures) -> np.ndarray:
        features.validate()
        vm = _l2_normalize(_pad(features.metadata, ALIGNED_DIM), self.eps)
        vc = _l2_normalize(_pad(features.code, ALIGNED_DIM), self.eps)
        vg = _l2_normalize(_pad(features.graph, ALIGNED_DIM), self.eps)
        matrix = np.stack([
            vm.reshape(32, 32),
            vc.reshape(32, 32),
            vg.reshape(32, 32),
        ], axis=0).astype(np.float32)
        if matrix.shape != MATRIX_SHAPE:
            raise RuntimeError(f"feature matrix produced {matrix.shape}, expected {MATRIX_SHAPE}")
        return matrix

    def generate(self, features: MultiViewFeatures) -> torch.Tensor:
        return torch.from_numpy(self.generate_numpy(features))
