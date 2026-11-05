from __future__ import annotations

from collections import defaultdict

from mvrfdet.core.constants import CENTRALITY_NAMES
from mvrfdet.core.types import APICallGraph


class CentralityAggregator:
    """Aggregate centrality by API name across a malicious package corpus.

    For each metric, sums all occurrences sharing an API name, then divides by
    the total number of malicious packages. The ranking score is the mean of the
    six corpus-averaged metrics, following the implementation-detail wording in
    the paper as closely as possible.
    """

    def __init__(self) -> None:
        self.package_count = 0
        self.sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.occurrences: dict[str, int] = defaultdict(int)
        self.packages_seen: dict[str, int] = defaultdict(int)

    def add_package(self, graph: APICallGraph, node_centralities: dict[str, dict[str, float]]) -> None:
        self.package_count += 1
        names_seen: set[str] = set()
        for node_id, node in graph.nodes.items():
            if node.kind != "api" or not node.api_name or node.api_name == "<dynamic-call>":
                continue
            metrics = node_centralities.get(node_id, {})
            for metric in CENTRALITY_NAMES:
                self.sums[node.api_name][metric] += float(metrics.get(metric, 1.0))
            self.occurrences[node.api_name] += 1
            names_seen.add(node.api_name)
        for name in names_seen:
            self.packages_seen[name] += 1

    def finalize(self) -> list[dict]:
        if self.package_count == 0:
            return []
        records = []
        for api_name, sums in self.sums.items():
            averages = {metric: float(sums[metric] / self.package_count) for metric in CENTRALITY_NAMES}
            score = sum(averages.values()) / len(CENTRALITY_NAMES)
            records.append({
                "api_name": api_name,
                "score": score,
                "centrality": averages,
                "occurrences": int(self.occurrences[api_name]),
                "packages_seen": int(self.packages_seen[api_name]),
                "corpus_packages": int(self.package_count),
            })
        records.sort(key=lambda x: (-x["score"], -x["occurrences"], x["api_name"]))
        return records
