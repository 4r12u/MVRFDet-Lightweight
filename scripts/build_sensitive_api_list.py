#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mvrfdet.offline.sensitive_api.builder import (
    build_candidates,
    sensitive_list_document,
    verify_candidates,
)
from mvrfdet.offline.sensitive_api.deepseek import DeepSeekVerifier
from mvrfdet.static_analysis.scalpel_adapter import StaticAnalysisAdapter


def discover_packages(root: Path) -> list[Path]:
    children = [p for p in sorted(root.iterdir()) if p.is_dir()]
    return children if children else [root]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline sensitive API knowledge base")
    parser.add_argument("--packages-dir", required=True, type=Path, help="Directory containing verified malicious package directories")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--contexts-per-api", type=int, default=3)
    parser.add_argument("--prompt", type=Path, default=Path("prompts/sensitive_api/paper_prompt.txt"))
    parser.add_argument("--backend", choices=["auto", "ast", "scalpel"], default="auto")
    parser.add_argument("--skip-llm", action="store_true", help="Only generate candidates_top500.json and contexts.json")
    parser.add_argument("--llm-api-key-env", default="DEEPSEEK_API_KEY", help="Environment variable containing the DeepSeek key")
    parser.add_argument("--llm-model", default=None, help="Model ID; defaults to DEEPSEEK_MODEL or deepseek-v4-pro")
    parser.add_argument("--llm-api-url", default=None, help="Optional Chat Completions endpoint")
    parser.add_argument("--llm-thinking", action="store_true", help="Enable DeepSeek reasoning mode")
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--llm-refresh", action="store_true")
    args = parser.parse_args()
    for field in ("top_k", "contexts_per_api", "llm_timeout", "llm_retries", "llm_max_tokens"):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if not args.skip_llm and not os.environ.get(args.llm_api_key_env):
        parser.error(f"Environment variable {args.llm_api_key_env} is not set")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    package_dirs = discover_packages(args.packages_dir)
    analyzer = StaticAnalysisAdapter(args.backend)
    candidates, contexts = build_candidates(package_dirs, analyzer, top_k=args.top_k, contexts_per_api=args.contexts_per_api)

    (args.output_dir / "candidates_top500.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "candidate_contexts.json").write_text(
        json.dumps(contexts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Generated {len(candidates)} candidates from {len(package_dirs)} packages")

    if args.skip_llm:
        return

    verifier = DeepSeekVerifier(
        api_key=os.environ[args.llm_api_key_env], api_url=args.llm_api_url,
        model=args.llm_model or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-pro",
        timeout=args.llm_timeout,
    )
    verified = verify_candidates(
        candidates=candidates,
        contexts=contexts,
        prompt_path=args.prompt,
        cache_dir=args.output_dir / "llm_results",
        verifier=verifier,
        thinking=args.llm_thinking,
        retries=args.llm_retries,
        max_tokens=args.llm_max_tokens,
        refresh=args.llm_refresh,
    )
    document = sensitive_list_document(verified, model=verifier.model)
    (args.output_dir / "sensitive_apis.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"LLM marked {document['count']} APIs as sensitive")


if __name__ == "__main__":
    main()
