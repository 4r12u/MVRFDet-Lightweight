#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mvrfdet.stage3.model import LightweightMVRFDet
from mvrfdet.training.dataset import CachedFeatureDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    loader = DataLoader(CachedFeatureDataset(args.manifest), batch_size=args.batch_size, shuffle=False)
    model = LightweightMVRFDet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
    model.eval()

    tp = fp = tn = fn = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(dim=1).cpu()
            for p, target in zip(pred.tolist(), y.tolist()):
                if p == 1 and target == 1: tp += 1
                elif p == 1 and target == 0: fp += 1
                elif p == 0 and target == 0: tn += 1
                else: fn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    print(f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} accuracy={accuracy:.4f}")
    print(f"tp={tp} fp={fp} tn={tn} fn={fn}")


if __name__ == "__main__":
    main()
