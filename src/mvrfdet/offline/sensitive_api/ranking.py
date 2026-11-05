from __future__ import annotations


def top_candidates(records: list[dict], top_k: int = 500) -> list[dict]:
    return records[:max(0, int(top_k))]
