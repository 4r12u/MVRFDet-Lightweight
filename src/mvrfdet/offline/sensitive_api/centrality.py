from __future__ import annotations

from mvrfdet.core.types import APICallGraph
from mvrfdet.graph_utils import compute_six_centralities


def centrality_for_api_graph(graph: APICallGraph) -> dict[str, dict[str, float]]:
    """Compute the six paper metrics and add 1 to every node value."""
    nx_graph = graph.to_networkx()
    raw = compute_six_centralities(nx_graph, add_one=True)
    result: dict[str, dict[str, float]] = {}
    for node_id, node in graph.nodes.items():
        result[node_id] = {metric: float(raw[metric].get(node_id, 1.0)) for metric in raw}
    return result
