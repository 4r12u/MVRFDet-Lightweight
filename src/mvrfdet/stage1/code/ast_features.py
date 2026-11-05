from __future__ import annotations

import ast
from collections import Counter

import numpy as np
from sklearn.feature_extraction import FeatureHasher


class _StatsVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.max_depth = 0
        self.current_depth = 0
        self.max_function_depth = 0
        self.function_depth = 0

    def generic_visit(self, node: ast.AST) -> None:
        self.counts[type(node).__name__] += 1
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        is_function = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        if is_function:
            self.function_depth += 1
            self.max_function_depth = max(self.max_function_depth, self.function_depth)
        super().generic_visit(node)
        if is_function:
            self.function_depth -= 1
        self.current_depth -= 1


def ast_feature_dict(source: str) -> dict[str, float]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"syntax_error": 1.0}
    visitor = _StatsVisitor()
    visitor.visit(tree)
    result: dict[str, float] = {f"node:{name}": float(count) for name, count in visitor.counts.items()}
    result.update(
        {
            "ast_total_nodes": float(sum(visitor.counts.values())),
            "ast_max_depth": float(visitor.max_depth),
            "function_nesting_depth": float(visitor.max_function_depth),
            "branch_count": float(visitor.counts["If"] + visitor.counts["IfExp"] + visitor.counts["Match"]),
            "loop_count": float(visitor.counts["For"] + visitor.counts["AsyncFor"] + visitor.counts["While"]),
            "exception_count": float(visitor.counts["Try"] + visitor.counts["Raise"] + visitor.counts["ExceptHandler"]),
            "call_count": float(visitor.counts["Call"]),
            "import_count": float(visitor.counts["Import"] + visitor.counts["ImportFrom"]),
            "function_count": float(visitor.counts["FunctionDef"] + visitor.counts["AsyncFunctionDef"] + visitor.counts["Lambda"]),
            "assignment_count": float(visitor.counts["Assign"] + visitor.counts["AnnAssign"] + visitor.counts["AugAssign"]),
            "cyclomatic_like": float(1 + visitor.counts["If"] + visitor.counts["For"] + visitor.counts["AsyncFor"] + visitor.counts["While"] + visitor.counts["ExceptHandler"] + visitor.counts["BoolOp"]),
        }
    )
    return result


class ASTFeatureEncoder:
    def __init__(self, dimension: int = 256):
        self.dimension = dimension
        self.hasher = FeatureHasher(n_features=dimension, input_type="dict", alternate_sign=False)

    def encode(self, source: str) -> np.ndarray:
        vec = self.hasher.transform([ast_feature_dict(source)]).toarray()[0]
        return vec.astype(np.float32)
