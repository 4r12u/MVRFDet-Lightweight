from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from pathlib import Path

from mvrfdet.static_analysis.scalpel_adapter import StaticAnalysisAdapter
from mvrfdet.stage1.code.context import extract_code_context
from mvrfdet.core.types import SuspiciousAPIOccurrence
from .aggregate import CentralityAggregator
from .centrality import centrality_for_api_graph
from .ranking import top_candidates
from .deepseek import DeepSeekVerifier, load_paper_prompt, render_prompt


def build_candidates(package_dirs: list[Path], analyzer: StaticAnalysisAdapter, top_k: int = 500, contexts_per_api: int = 3) -> tuple[list[dict], dict[str, list[str]]]:
    aggregator = CentralityAggregator()
    contexts: dict[str, list[str]] = defaultdict(list)

    for package_dir in package_dirs:
        graph = analyzer.extract_api_call_graph(package_dir)
        centralities = centrality_for_api_graph(graph)
        aggregator.add_package(graph, centralities)
        for node in graph.nodes.values():
            if node.kind != "api" or len(contexts[node.api_name]) >= contexts_per_api:
                continue
            file_path = Path(node.file)
            if not file_path.exists():
                continue
            occurrence = SuspiciousAPIOccurrence(node.id, node.api_name, file_path, node.line, node.column, node.function)
            try:
                snippet = extract_code_context(occurrence)
            except Exception:
                continue
            if snippet.text and snippet.text not in contexts[node.api_name]:
                contexts[node.api_name].append(snippet.text)

    candidates = top_candidates(aggregator.finalize(), top_k=top_k)
    candidate_names = {item["api_name"] for item in candidates}
    return candidates, {name: contexts.get(name, []) for name in candidate_names}


def verify_candidates(
    candidates: list[dict],
    contexts: dict[str, list[str]],
    prompt_path: Path,
    cache_dir: Path,
    verifier: DeepSeekVerifier,
    *,
    thinking: bool = False,
    retries: int = 3,
    max_tokens: int = 4096,
    refresh: bool = False,
) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    template = load_paper_prompt(prompt_path)
    endpoint_hash = hashlib.sha256(verifier.api_url.encode("utf-8")).hexdigest()
    verified: list[dict] = []
    for rank, candidate in enumerate(candidates, 1):
        api_name = candidate["api_name"]
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in api_name)[:180]
        cache_path = cache_dir / f"{rank:04d}_{safe_name}.json"
        prompt = render_prompt(template, api_name, contexts.get(api_name, []))
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        same_case = (
            cached.get("candidate", {}).get("api_name") == api_name
            and cached.get("model") == verifier.model
            and cached.get("prompt_sha256") == prompt_hash
            and cached.get("thinking") == thinking
            and cached.get("max_tokens") == max_tokens
            and cached.get("endpoint_sha256") == endpoint_hash
        )
        if same_case and not refresh:
            result = cached
        else:
            response = verifier.verify(prompt, retries=retries, thinking=thinking, max_tokens=max_tokens)
            result = {
                "candidate_rank": rank,
                "candidate": candidate,
                "context_count": len(contexts.get(api_name, [])),
                "prompt_sha256": prompt_hash,
                "thinking": thinking,
                "max_tokens": max_tokens,
                "endpoint_sha256": endpoint_hash,
                "model": response["model"],
                "raw_response": response["raw_response"],
                "parsed": response["parsed"],
                "reasoning_content": response.get("reasoning_content"),
                "finish_reason": response.get("finish_reason"),
                "usage": response.get("usage"),
            }
            cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        parsed = result.get("parsed", {})
        value = parsed.get("is_potentially_malicious", False)
        is_sensitive = value is True or str(value).strip().lower() == "true"
        if is_sensitive:
            verified.append(result)
    return verified


def sensitive_list_document(verified: list[dict], model: str) -> dict:
    apis = []
    for item in verified:
        name = item.get("parsed", {}).get("api_name") or item.get("candidate", {}).get("api_name")
        if name:
            apis.append(str(name))
    return {
        "version": "v1",
        "source": "six-centralities+paper-prompt+deepseek",
        "llm_model": model,
        "count": len(sorted(set(apis))),
        "apis": sorted(set(apis)),
    }
