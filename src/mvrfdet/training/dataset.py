from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from mvrfdet.core.types import MultiViewFeatures
from mvrfdet.stage2.feature_matrix import FeatureMatrixGenerator


class CachedFeatureDataset(Dataset):
    """CSV manifest dataset. Required columns: feature_path,label."""

    def __init__(self, manifest: str | Path):
        self.manifest = Path(manifest)
        with self.manifest.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Empty manifest: {manifest}")
        missing = {"feature_path", "label"} - set(rows[0])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        self.rows = rows
        self.matrix_generator = FeatureMatrixGenerator()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = Path(row["feature_path"])
        if not path.is_absolute():
            path = (self.manifest.parent / path).resolve()
        features = MultiViewFeatures.load_npz(path)
        matrix = self.matrix_generator.generate(features)
        label = int(row["label"])
        return matrix, torch.tensor(label, dtype=torch.long)
