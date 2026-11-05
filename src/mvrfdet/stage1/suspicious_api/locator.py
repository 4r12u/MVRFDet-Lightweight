from __future__ import annotations

import json
from pathlib import Path

from mvrfdet.core.types import APICallGraph, SuspiciousAPIOccurrence


def load_sensitive_api_list(path: str | Path) -> set[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        apis = data
    elif isinstance(data, dict):
        apis = data.get("apis", [])
    else:
        raise ValueError("Sensitive API file must contain a list or an object with an 'apis' list")
    return {str(api).strip() for api in apis if str(api).strip()}


def _matches(api_name: str, sensitive: set[str]) -> bool:
    if api_name in sensitive:
        return True
    # Support locally imported aliases after static analysis loses a module prefix.
    # Exact matching remains preferred to avoid broad substring matching.
    short = api_name.split(".")[-1]
    candidates = {item for item in sensitive if item.split(".")[-1] == short}
    return len(candidates) == 1


def locate_suspicious_apis(graph: APICallGraph, sensitive_apis: set[str]) -> list[SuspiciousAPIOccurrence]:
    found: list[SuspiciousAPIOccurrence] = []
    for node in graph.nodes.values():
        if node.kind != "api" or not _matches(node.api_name, sensitive_apis):
            continue
        found.append(
            SuspiciousAPIOccurrence(
                node_id=node.id,
                api_name=node.api_name,
                file=Path(node.file),
                line=node.line,
                column=node.column,
                function=node.function,
            )
        )
    return sorted(found, key=lambda x: (str(x.file), x.line, x.column, x.api_name))
