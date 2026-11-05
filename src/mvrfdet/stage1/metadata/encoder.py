from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from mvrfdet.core.constants import METADATA_DIM
from .features import handcrafted_metadata_features, metadata_text


class StaticMetadataEncoder:
    """Lightweight replacement for all-mpnet-base-v2.

    32 handcrafted fields + 736 hashed word/ngram dimensions = Vm[768].
    HashingVectorizer is stateless, so no fit/training corpus is required.
    """

    def __init__(self, dimension: int = METADATA_DIM, handcrafted_dimension: int = 32):
        if dimension <= handcrafted_dimension:
            raise ValueError("dimension must exceed handcrafted dimension")
        self.dimension = dimension
        self.handcrafted_dimension = handcrafted_dimension
        self.hashed_dimension = dimension - handcrafted_dimension
        self.vectorizer = HashingVectorizer(
            n_features=self.hashed_dimension,
            alternate_sign=False,
            norm=None,
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[\w@./:+-]+\b",
        )

    def encode(self, metadata: dict[str, str]) -> np.ndarray:
        manual = handcrafted_metadata_features(metadata)
        if manual.shape != (self.handcrafted_dimension,):
            raise RuntimeError(f"Expected {self.handcrafted_dimension} handcrafted metadata features")
        hashed = self.vectorizer.transform([metadata_text(metadata)]).toarray()[0].astype(np.float32)
        vector = np.concatenate([manual, hashed], dtype=np.float32)
        if vector.shape != (self.dimension,):
            raise RuntimeError(f"metadata encoder produced {vector.shape}, expected {(self.dimension,)}")
        return vector
