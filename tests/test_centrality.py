from mvrfdet.core.types import APICallGraph, APINode
from mvrfdet.offline.sensitive_api.centrality import centrality_for_api_graph
from mvrfdet.offline.sensitive_api.aggregate import CentralityAggregator


def test_centrality_aggregation():
    graph = APICallGraph(backend="test")
    graph.add_node(APINode("a", "os.system", "a.py", 1, 0))
    graph.add_node(APINode("b", "socket.socket", "a.py", 2, 0))
    graph.add_node(APINode("c", "os.system", "a.py", 3, 0))
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    values = centrality_for_api_graph(graph)
    assert all(v >= 1.0 for metrics in values.values() for v in metrics.values())
    agg = CentralityAggregator()
    agg.add_package(graph, values)
    records = agg.finalize()
    assert records
    names = {r["api_name"] for r in records}
    assert "os.system" in names
