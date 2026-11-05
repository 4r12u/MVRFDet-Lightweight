from __future__ import annotations

import ast
from pathlib import Path

from mvrfdet.core.types import APICallGraph, APINode


def _qualified_name(node: ast.AST, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _qualified_name(node.value, aliases)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _qualified_name(node.func, aliases)
    return ""


class _CallCollector(ast.NodeVisitor):
    def __init__(self, file_path: Path, source: str):
        self.file_path = file_path
        self.source = source
        self.aliases: dict[str, str] = {}
        self.function_stack: list[str] = ["<module>"]
        self.calls_by_scope: dict[str, list[APINode]] = {"<module>": []}
        self.counter = 0

    @property
    def scope(self) -> str:
        return ".".join(self.function_stack[1:]) if len(self.function_stack) > 1 else "<module>"

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.aliases[alias.asname or alias.name.split(".")[0]] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            full = f"{module}.{alias.name}" if module else alias.name
            self.aliases[alias.asname or alias.name] = full

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name)
        self.calls_by_scope.setdefault(self.scope, [])
        for stmt in node.body:
            self.visit(stmt)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.function_stack.append(f"<lambda@{getattr(node, 'lineno', 0)}>")
        self.calls_by_scope.setdefault(self.scope, [])
        self.visit(node.body)
        self.function_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        name = _qualified_name(node.func, self.aliases) or "<dynamic-call>"
        self.counter += 1
        node_id = f"{self.file_path}:{getattr(node, 'lineno', 0)}:{getattr(node, 'col_offset', 0)}:{self.counter}"
        api_node = APINode(
            id=node_id,
            api_name=name,
            file=str(self.file_path),
            line=int(getattr(node, "lineno", 0) or 0),
            column=int(getattr(node, "col_offset", 0) or 0),
            function=self.scope,
            kind="api",
        )
        self.calls_by_scope.setdefault(self.scope, []).append(api_node)
        # Keep nested calls too (e.g. foo(bar())).
        self.generic_visit(node)


def extract_ast_api_call_graph(package_path: str | Path) -> APICallGraph:
    """Build a lightweight API-call-occurrence graph from Python ASTs.

    This is a runnable fallback, not a claim of Scalpel equivalence. Nodes are
    API call occurrences. Edges connect calls in source order within the same
    lexical function/module and add lightweight interprocedural links when a
    call resolves to a local function name.
    """
    root = Path(package_path)
    graph = APICallGraph(backend="ast-fallback")
    all_scopes: dict[str, list[APINode]] = {}
    local_function_entries: dict[str, APINode] = {}

    for py_file in sorted(root.rglob("*.py")):
        try:
            source = py_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, OSError):
            continue
        collector = _CallCollector(py_file, source)
        collector.visit(tree)
        for scope, calls in collector.calls_by_scope.items():
            key = f"{py_file}:{scope}"
            ordered = sorted(calls, key=lambda n: (n.line, n.column, n.id))
            all_scopes[key] = ordered
            if scope != "<module>" and ordered:
                local_function_entries.setdefault(scope.split(".")[-1], ordered[0])
            for node in ordered:
                graph.add_node(node)
            for left, right in zip(ordered, ordered[1:]):
                graph.add_edge(left.id, right.id)

    # Small interprocedural approximation: a call named foo() points to the
    # first API call inside local function foo when known.
    for node in list(graph.nodes.values()):
        short_name = node.api_name.split(".")[-1]
        target = local_function_entries.get(short_name)
        if target is not None and target.id != node.id:
            graph.add_edge(node.id, target.id)

    return graph
