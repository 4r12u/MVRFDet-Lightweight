#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mvrfdet.pipeline import LightweightFeaturePipeline
from mvrfdet.offline.sensitive_api.package_reasoning import (
    add_package_reasoning_arguments,
    options_from_args,
    run_package_reasoning,
)
from mvrfdet.offline.sensitive_api.deepseek import DeepSeekVerificationError
from mvrfdet.reporting import write_package_report
from mvrfdet.stage1.suspicious_api.locator import load_sensitive_api_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Vm/Vc/Vg from one unpacked PyPI package")
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--sensitive-list", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Output .npz")
    parser.add_argument("--report-dir", type=Path, help="Per-package stage outputs and graph visualizations")
    parser.add_argument("--checkpoint", type=Path, help="Optional trained checkpoint for Stage 3 inference")
    parser.add_argument("--backend", choices=["auto", "ast", "scalpel"], default="auto")
    add_package_reasoning_arguments(parser)
    args = parser.parse_args()
    if args.output is None and args.report_dir is None:
        parser.error("Provide --output, --report-dir, or both")
    if args.checkpoint is not None and args.report_dir is None:
        parser.error("--checkpoint requires --report-dir")
    if args.llm_reasoning and args.report_dir is None:
        parser.error("--llm-reasoning requires --report-dir")
    try:
        llm_options = options_from_args(args) if args.llm_reasoning else None
    except (ValueError, OSError, DeepSeekVerificationError) as exc:
        parser.error(str(exc))

    sensitive = load_sensitive_api_list(args.sensitive_list)
    pipeline = LightweightFeaturePipeline(static_backend=args.backend)
    artifacts = pipeline.extract_with_artifacts(args.package, sensitive)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        artifacts.features.save_npz(args.output)
        diagnostics_path = args.output.with_suffix(args.output.suffix + ".json")
        diagnostics_path.write_text(json.dumps(artifacts.features.diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.report_dir is not None:
        result = write_package_report(args.report_dir, args.package.name, args.package, artifacts, args.checkpoint)
        if llm_options is not None:
            result.update(run_package_reasoning(artifacts, sensitive, args.report_dir, llm_options))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(artifacts.features.diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
