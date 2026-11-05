import numpy as np
import torch

from mvrfdet.core.types import MultiViewFeatures
from mvrfdet.stage2.feature_matrix import FeatureMatrixGenerator
from mvrfdet.stage3.model import LightweightMVRFDet


def test_feature_matrix_and_model_shapes():
    features = MultiViewFeatures(
        metadata=np.arange(768, dtype=np.float32),
        code=np.ones(768, dtype=np.float32),
        graph=np.ones(1024, dtype=np.float32),
    )
    matrix = FeatureMatrixGenerator().generate(features)
    assert tuple(matrix.shape) == (3, 32, 32)
    # Padding remains exactly zero after view-wise L2 normalization.
    assert torch.all(matrix[0].reshape(-1)[768:] == 0)
    model = LightweightMVRFDet()
    logits, attention = model(matrix.unsqueeze(0), return_attention=True)
    assert tuple(logits.shape) == (1, 2)
    assert tuple(attention.shape) == (1, 256)
