from __future__ import annotations

import math
from statistics import mean

import networkx as nx
import numpy as np

from mvrfdet.core.constants import CENTRALITY_NAMES


def _safe_float(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return value


def _zero_map(graph: nx.Graph) -> dict[object, float]:
    return {node: 0.0 for node in graph.nodes}


def compute_six_centralities(graph: nx.Graph, add_one: bool = False) -> dict[str, dict[object, float]]:
    """Compute the six paper centralities with defensive fallbacks.

    Katz alpha is chosen from the observed spectral radius because the paper
    constrains alpha but does not publish a concrete value. This is marked as
    an engineering choice in REPRODUCTION_GAPS.md.
    """
    if graph.number_of_nodes() == 0:
        return {name: {} for name in CENTRALITY_NAMES}
    if graph.number_of_nodes() == 1:
        base = {node: 0.0 for node in graph.nodes}
        result = {name: dict(base) for name in CENTRALITY_NAMES}
    else:
        result: dict[str, dict[object, float]] = {}
        try:
            result["degree"] = {k: _safe_float(v) for k, v in nx.degree_centrality(graph).items()}
        except Exception:
            result["degree"] = _zero_map(graph)
        try:
            result["closeness"] = {k: _safe_float(v) for k, v in nx.closeness_centrality(graph).items()}
        except Exception:
            result["closeness"] = _zero_map(graph)
        try:
            result["harmonic"] = {k: _safe_float(v) for k, v in nx.harmonic_centrality(graph).items()}
        except Exception:
            result["harmonic"] = _zero_map(graph)

        # Katz: alpha must be < 1/lambda_max. Work on a float adjacency matrix.
        try:
            nodes = list(graph.nodes)
            A = nx.to_numpy_array(graph, nodelist=nodes, dtype=float)
            eigvals = np.linalg.eigvals(A)
            spectral_radius = max((abs(complex(x)) for x in eigvals), default=0.0)
            alpha = 0.1 if spectral_radius <= 0 else min(0.1, 0.9 / float(spectral_radius))
            katz = nx.katz_centrality_numpy(graph, alpha=alpha, beta=1.0, normalized=True)
            result["katz"] = {k: _safe_float(v) for k, v in katz.items()}
        except Exception:
            result["katz"] = _zero_map(graph)
        try:
            # numpy implementation is deterministic and avoids convergence knobs.
            eigen = nx.eigenvector_centrality_numpy(graph)
            result["eigenvector"] = {k: _safe_float(v) for k, v in eigen.items()}
        except Exception:
            result["eigenvector"] = _zero_map(graph)
        try:
            result["betweenness"] = {k: _safe_float(v) for k, v in nx.betweenness_centrality(graph, normalized=True).items()}
        except Exception:
            result["betweenness"] = _zero_map(graph)

    if add_one:
        for metric in CENTRALITY_NAMES:
            # Numerical eigensolvers can return a tiny negative value for a
            # theoretically non-negative centrality. Keep the paper's >= 1
            # offline ranking contract after the +1 shift.
            result[metric] = {node: max(0.0, value) + 1.0 for node, value in result[metric].items()}
    return result


def distribution_stats(values) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
    }


def graph_statistics(graph: nx.Graph, sensitive_apis: set[str] | None = None, prefix: str = "graph") -> dict[str, float]:
    sensitive_apis = sensitive_apis or set()
    n = graph.number_of_nodes()
    e = graph.number_of_edges()
    stats: dict[str, float] = {
        f"{prefix}:nodes": float(n),
        f"{prefix}:edges": float(e),
        f"{prefix}:density": float(nx.density(graph)) if n > 1 else 0.0,
        f"{prefix}:self_loops": float(nx.number_of_selfloops(graph)),
        f"{prefix}:is_directed": float(graph.is_directed()),
    }
    if n == 0:
        return stats

    directed = graph if graph.is_directed() else nx.DiGraph(graph)
    undirected = graph.to_undirected() if graph.is_directed() else graph
    try:
        stats[f"{prefix}:components"] = float(nx.number_connected_components(undirected))
    except Exception:
        stats[f"{prefix}:components"] = 0.0
    try:
        stats[f"{prefix}:weak_components"] = float(nx.number_weakly_connected_components(directed))
        stats[f"{prefix}:strong_components"] = float(nx.number_strongly_connected_components(directed))
    except Exception:
        stats[f"{prefix}:weak_components"] = stats.get(f"{prefix}:components", 0.0)
        stats[f"{prefix}:strong_components"] = 0.0
    try:
        stats[f"{prefix}:is_dag"] = float(nx.is_directed_acyclic_graph(directed))
    except Exception:
        stats[f"{prefix}:is_dag"] = 0.0
    try:
        stats[f"{prefix}:transitivity"] = float(nx.transitivity(undirected))
        stats[f"{prefix}:avg_clustering"] = float(nx.average_clustering(undirected)) if n > 1 else 0.0
    except Exception:
        stats[f"{prefix}:transitivity"] = 0.0
        stats[f"{prefix}:avg_clustering"] = 0.0

    for label, values in (
        ("degree", [d for _, d in graph.degree()]),
        ("in_degree", [d for _, d in directed.in_degree()]),
        ("out_degree", [d for _, d in directed.out_degree()]),
    ):
        for name, value in distribution_stats(values).items():
            stats[f"{prefix}:{label}:{name}"] = value

    centralities = compute_six_centralities(graph, add_one=False)
    for metric, mapping in centralities.items():
        for name, value in distribution_stats(mapping.values()).items():
            stats[f"{prefix}:centrality:{metric}:{name}"] = value

    # Security-specific API occurrence signals when api_name is available.
    sensitive_nodes = []
    categories: dict[str, int] = {}
    for node, attrs in graph.nodes(data=True):
        api_name = str(attrs.get("api_name", ""))
        if api_name in sensitive_apis or (api_name and len({x for x in sensitive_apis if x.split('.')[-1] == api_name.split('.')[-1]}) == 1):
            sensitive_nodes.append(node)
    stats[f"{prefix}:sensitive_count"] = float(len(sensitive_nodes))
    stats[f"{prefix}:sensitive_ratio"] = float(len(sensitive_nodes) / n) if n else 0.0
    sensitive_set = set(sensitive_nodes)
    stats[f"{prefix}:sensitive_edges"] = float(sum(1 for a, b in graph.edges() if a in sensitive_set and b in sensitive_set))

    # Bounded path statistics on largest component to avoid cubic work on huge graphs.
    try:
        components = list(nx.connected_components(undirected))
        largest = max(components, key=len) if components else set()
        if largest:
            sample_nodes = sorted(largest, key=str)[:300]
            sub = undirected.subgraph(sample_nodes)
            if sub.number_of_nodes() > 1 and nx.is_connected(sub):
                stats[f"{prefix}:avg_shortest_path"] = float(nx.average_shortest_path_length(sub))
                stats[f"{prefix}:diameter"] = float(nx.diameter(sub))
            else:
                stats[f"{prefix}:avg_shortest_path"] = 0.0
                stats[f"{prefix}:diameter"] = 0.0
    except Exception:
        stats[f"{prefix}:avg_shortest_path"] = 0.0
        stats[f"{prefix}:diameter"] = 0.0

    return stats
