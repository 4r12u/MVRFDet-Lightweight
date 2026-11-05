from __future__ import annotations

import importlib.util
from pathlib import Path

from mvrfdet.core.types import APICallGraph
from .ast_call_graph import extract_ast_api_call_graph


class ScalpelIntegrationError(RuntimeError):
    pass


class StaticAnalysisAdapter:
    """Static-analysis façade.

    `backend="ast"` is fully runnable in this repository. `backend="scalpel"`
    is intentionally a strict integration boundary: the paper specifies Scalpel,
    but Scalpel is not bundled here and its concrete output adapter must be
    implemented against the exact installed version. `backend="auto"` uses
    Scalpel only after such an adapter is supplied; otherwise it uses AST.
    """

    def __init__(self, backend: str = "auto"):
        if backend not in {"auto", "ast", "scalpel"}:
            raise ValueError("backend must be one of: auto, ast, scalpel")
        self.backend = backend

    @staticmethod
    def scalpel_available() -> bool:
        return importlib.util.find_spec("scalpel") is not None

    def extract_api_call_graph(self, package_path: str | Path) -> APICallGraph:
        if self.backend == "ast":
            return extract_ast_api_call_graph(package_path)
        if self.backend == "scalpel":
            return self._extract_with_scalpel(package_path)
        # auto: keep the runnable path deterministic until a version-specific
        # Scalpel adapter is implemented.
        return extract_ast_api_call_graph(package_path)

    def _extract_with_scalpel(self, package_path: str | Path) -> APICallGraph:
        if not self.scalpel_available():
            raise ScalpelIntegrationError(
                "Scalpel is not installed. Install the paper-compatible Scalpel version "
                "and implement the version-specific conversion in scalpel_adapter.py."
            )
        raise ScalpelIntegrationError(
            "Scalpel is installed, but a version-specific graph conversion is still required. "
            "The repository deliberately does not guess an undocumented Scalpel API/output schema."
        )
