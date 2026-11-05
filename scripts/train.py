#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mvrfdet.stage3.model import LightweightMVRFDet
from mvrfdet.training.dataset import CachedFeatureDataset
from mvrfdet.training.engine import run_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_ds = CachedFeatureDataset(args.train_manifest)
    val_ds = CachedFeatureDataset(args.val_manifest)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = LightweightMVRFDet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss = float("inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics.loss:.5f} train_acc={train_metrics.accuracy:.4f} "
            f"val_loss={val_metrics.loss:.5f} val_acc={val_metrics.accuracy:.4f}"
        )
        if val_metrics.loss < best_loss:
            best_loss = val_metrics.loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": best_loss}, args.output)


if __name__ == "__main__":
    main()
