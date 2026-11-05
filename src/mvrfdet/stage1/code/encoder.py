from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from mvrfdet.core.constants import CODE_DIM
from mvrfdet.core.types import CodeContextSnippet
from .api_features import APIFeatureEncoder
from .ast_features import ASTFeatureEncoder


class StaticCodeEncoder:
    """AST + API + token-hash replacement for LongCoder.

    Each snippet becomes 768 dimensions (256/256/256), then package-level
    mean pooling mirrors the paper's snippet aggregation role.
    """

    def __init__(self, dimension: int = CODE_DIM):
        if dimension != 768:
            raise ValueError("Current lightweight code encoder is defined for 768 dimensions")
        self.dimension = dimension
        self.ast_encoder = ASTFeatureEncoder(256)
        self.api_encoder = APIFeatureEncoder(256)
        self.token_vectorizer = HashingVectorizer(
            n_features=256,
            alternate_sign=False,
            norm=None,
            analyzer="word",
            lowercase=False,
            ngram_range=(1, 3),
            token_pattern=r"(?u)\b[A-Za-z_][A-Za-z0-9_\.]*\b|\b\d+\b",
        )

    @staticmethod
    def _normalize_tokens(source: str) -> str:
        # Preserve API-like identifiers while suppressing literal payload detail.
        source = re.sub(r"(?s)([rubfRUBF]*)(['\"]).*?\2", " <STR> ", source)
        source = re.sub(r"\b\d+(?:\.\d+)?\b", " <NUM> ", source)
        return source

    def encode_snippet(self, source: str) -> np.ndarray:
        ast_vec = self.ast_encoder.encode(source)
        api_vec = self.api_encoder.encode(source)
        token_vec = self.token_vectorizer.transform([self._normalize_tokens(source)]).toarray()[0].astype(np.float32)
        vector = np.concatenate([ast_vec, api_vec, token_vec]).astype(np.float32)
        if vector.shape != (self.dimension,):
            raise RuntimeError(f"code encoder produced {vector.shape}")
        return vector

    def encode(self, snippets: list[CodeContextSnippet] | list[str]) -> np.ndarray:
        if not snippets:
            return np.zeros(self.dimension, dtype=np.float32)
        vectors: list[np.ndarray] = []
        for snippet in snippets:
            text = snippet.text if isinstance(snippet, CodeContextSnippet) else str(snippet)
            vectors.append(self.encode_snippet(text))
        return np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
