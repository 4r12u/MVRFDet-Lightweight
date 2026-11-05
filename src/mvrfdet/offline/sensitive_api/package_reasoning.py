"""Optional DeepSeek reasoning over suspicious APIs in one package report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from mvrfdet.pipeline import PipelineArtifacts
from mvrfdet.reporting import save_llm_case
from mvrfdet.stage1.code.context import extract_code_context

from .deepseek import DeepSeekVerificationError, DeepSeekVerifier, load_paper_prompt, render_prompt


@dataclass(frozen=True)
class PackageReasoningOptions:
    key_env: str
    model: str
    api_url: str | None
    prompt_path: Path
    api_name: str | None
    max_apis: int
    contexts_per_api: int
    timeout: int
    retries: int
    max_tokens: int
    refresh: bool


def add_package_reasoning_arguments(parser: argparse.ArgumentParser, *, include_enable_flag: bool = True) -> None:
    if include_enable_flag:
        parser.add_argument("--llm-reasoning", action="store_true", help="Verify suspicious APIs with DeepSeek thinking mode")
    parser.add_argument("--llm-api-key-env", default="DEEPSEEK_API_KEY", help="Environment variable containing the DeepSeek key")
    parser.add_argument("--llm-model", default=None, help="DeepSeek model; defaults to DEEPSEEK_MODEL or deepseek-v4-pro")
    parser.add_argument("--llm-api-url", default=None, help="Optional Chat Completions endpoint")
    parser.add_argument("--llm-api", default=None, help="Verify only this detected API name")
    parser.add_argument("--llm-max-apis", type=int, default=1, help="Maximum APIs to verify per package")
    parser.add_argument("--llm-contexts-per-api", type=int, default=2, help="Maximum code contexts per API")
    parser.add_argument("--llm-prompt", type=Path, default=Path("prompts/sensitive_api/paper_prompt.txt"))
    parser.add_argument("--llm-timeout", type=int, default=120, help="Timeout in seconds per request")
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--llm-max-tokens", type=int, default=4096)
    parser.add_argument("--llm-refresh", action="store_true", help="Re-query even when the same case is cached")


def options_from_args(args: argparse.Namespace) -> PackageReasoningOptions:
    for field in ("llm_max_apis", "llm_contexts_per_api", "llm_timeout", "llm_retries", "llm_max_tokens"):
        if getattr(args, field) < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.llm_api_key_env):
        raise ValueError("--llm-api-key-env must be an environment variable name")
    if not os.environ.get(args.llm_api_key_env):
        raise DeepSeekVerificationError(f"Environment variable {args.llm_api_key_env} is not set")
    if not args.llm_prompt.is_file():
        raise ValueError(f"Prompt file does not exist: {args.llm_prompt}")
    return PackageReasoningOptions(
        key_env=args.llm_api_key_env,
        model=args.llm_model or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-pro",
        api_url=args.llm_api_url or os.environ.get("DEEPSEEK_API_URL"),
        prompt_path=args.llm_prompt,
        api_name=args.llm_api,
        max_apis=args.llm_max_apis,
        contexts_per_api=args.llm_contexts_per_api,
        timeout=args.llm_timeout,
        retries=args.llm_retries,
        max_tokens=args.llm_max_tokens,
        refresh=args.llm_refresh,
    )


def run_package_reasoning(
    artifacts: PipelineArtifacts,
    sensitive_apis: set[str],
    report_dir: str | Path,
    options: PackageReasoningOptions,
) -> dict[str, object]:
    """Verify selected APIs and cache by model, API, and rendered prompt."""
    root = Path(report_dir)
    old_path = root / "stage0" / "llm_reasoning.json"
    old = json.loads(old_path.read_text(encoding="utf-8")) if old_path.exists() else {}
    old_cases = old.get("cases", []) if isinstance(old, dict) else []
    old_by_name = {item.get("api_name"): item for item in old_cases}

    names = {item.api_name for item in artifacts.suspicious}
    ordered = sorted(names, key=lambda name: (name not in sensitive_apis, name))
    if options.api_name is not None:
        ordered = [options.api_name] if options.api_name in names else []
    ordered = ordered[:options.max_apis]

    template = load_paper_prompt(options.prompt_path)
    verifier = DeepSeekVerifier(
        api_key=os.environ[options.key_env], api_url=options.api_url,
        model=options.model, timeout=options.timeout,
    )
    cases = []
    cached_count = 0
    endpoint_hash = hashlib.sha256(verifier.api_url.encode("utf-8")).hexdigest()
    for name in ordered:
        contexts = []
        digests = set()
        for occurrence in artifacts.suspicious:
            if occurrence.api_name != name or len(contexts) >= options.contexts_per_api:
                continue
            snippet = extract_code_context(occurrence)
            if snippet.digest not in digests:
                contexts.append(snippet.text)
                digests.add(snippet.digest)
        prompt = render_prompt(template, name, contexts)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cached = old_by_name.get(name, {})
        same_case = (
            cached.get("status") == "completed"
            and cached.get("model") == options.model
            and cached.get("prompt_sha256") == prompt_hash
            and cached.get("max_tokens") == options.max_tokens
            and cached.get("endpoint_sha256") == endpoint_hash
        )
        if same_case and not options.refresh:
            cases.append(cached)
            cached_count += 1
            continue
        record = {
            "api_name": name, "context_count": len(contexts), "prompt_sha256": prompt_hash,
            "model": options.model, "max_tokens": options.max_tokens,
            "endpoint_sha256": endpoint_hash,
        }
        try:
            response = verifier.verify(prompt, retries=options.retries, thinking=True, max_tokens=options.max_tokens)
            record.update({"status": "completed", **response})
        except DeepSeekVerificationError as exc:
            record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        cases.append(record)

    if not ordered:
        status = "no_matching_api" if options.api_name else "no_suspicious_api"
    elif all(case["status"] == "completed" for case in cases):
        status = "completed"
    else:
        status = "partial_error"
    result = {
        "status": status, "model": options.model, "key_env": options.key_env,
        "requested_api": options.api_name, "selected_api_count": len(ordered),
        "cached_count": cached_count,
        "parameters": {
            "endpoint_sha256": endpoint_hash,
            "prompt_path": str(options.prompt_path),
            "max_apis": options.max_apis,
            "contexts_per_api": options.contexts_per_api,
            "timeout_seconds": options.timeout,
            "retries": options.retries,
            "max_tokens": options.max_tokens,
            "refresh": options.refresh,
        },
        "cases": cases,
    }
    path = save_llm_case(root, result)
    return {"llm_status": status, "llm_cases": len(cases), "llm_cached": cached_count, "llm_result": str(path)}
