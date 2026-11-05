#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mvrfdet.pipeline import LightweightFeaturePipeline
from mvrfdet.stage1.suspicious_api.locator import load_sensitive_api_list
from mvrfdet.stage2.feature_matrix import FeatureMatrixGenerator
from mvrfdet.stage3.model import LightweightMVRFDet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--sensitive-list", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--backend", choices=["auto", "ast", "scalpel"], default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    sensitive = load_sensitive_api_list(args.sensitive_list)
    features = LightweightFeaturePipeline(static_backend=args.backend).extract(args.package, sensitive)
    matrix = FeatureMatrixGenerator().generate(features).unsqueeze(0)

    device = torch.device(args.device)
    model = LightweightMVRFDet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
    model.eval()
    with torch.no_grad():
        logits, attention = model(matrix.to(device), return_attention=True)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
    result = {
        "prediction": "malicious" if probabilities[1] >= probabilities[0] else "benign",
        "probabilities": {"benign": probabilities[0], "malicious": probabilities[1]},
        "diagnostics": features.diagnostics,
        "attention_mean": float(attention.mean().cpu()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
