from __future__ import annotations

import ast
import itertools

import networkx as nx


class ASTCFGBuilder:
    """Approximate CFG builder used by the lightweight implementation.

    It preserves sequential flow, branch splits/merges, and loop back-edges.
    It is intentionally simple and replaceable with a Scalpel CFG adapter.
    """

    def __init__(self) -> None:
        self._counter = itertools.count()
        self.graph = nx.DiGraph()

    def _add_stmt(self, stmt: ast.stmt) -> str:
        node_id = f"n{next(self._counter)}"
        self.graph.add_node(
            node_id,
            stmt_type=type(stmt).__name__,
            line=int(getattr(stmt, "lineno", 0) or 0),
        )
        return node_id

    def _build_block(self, statements: list[ast.stmt], incoming: list[str] | None = None) -> list[str]:
        exits = list(incoming or [])
        first = True
        for stmt in statements:
            current = self._add_stmt(stmt)
            if exits:
                for prev in exits:
                    self.graph.add_edge(prev, current)
            elif first:
                self.graph.nodes[current]["entry"] = True
            first = False

            if isinstance(stmt, ast.If):
                true_exits = self._build_block(stmt.body, [current]) if stmt.body else [current]
                false_exits = self._build_block(stmt.orelse, [current]) if stmt.orelse else [current]
                exits = list(dict.fromkeys(true_exits + false_exits))
            elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                body_exits = self._build_block(stmt.body, [current]) if stmt.body else [current]
                for tail in body_exits:
                    self.graph.add_edge(tail, current)
                orelse_exits = self._build_block(stmt.orelse, [current]) if stmt.orelse else [current]
                exits = list(dict.fromkeys([current] + orelse_exits))
            elif isinstance(stmt, ast.Try):
                branches: list[str] = []
                branches += self._build_block(stmt.body, [current]) if stmt.body else [current]
                for handler in stmt.handlers:
                    branches += self._build_block(handler.body, [current]) if handler.body else [current]
                if stmt.orelse:
                    branches = self._build_block(stmt.orelse, branches)
                if stmt.finalbody:
                    branches = self._build_block(stmt.finalbody, branches)
                exits = list(dict.fromkeys(branches))
            elif isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                self.graph.nodes[current]["terminal"] = True
                exits = []
            else:
                exits = [current]
        return exits

    def build(self, source: str) -> nx.DiGraph:
        self.graph = nx.DiGraph()
        self._counter = itertools.count()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self.graph
        exits = self._build_block(tree.body)
        for node in exits:
            if node in self.graph:
                self.graph.nodes[node]["exit"] = True
        if self.graph.number_of_nodes() == 1:
            only = next(iter(self.graph.nodes))
            self.graph.add_edge(only, only)
        return self.graph


def build_cfg(source: str) -> nx.DiGraph:
    return ASTCFGBuilder().build(source)
