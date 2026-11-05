"""Per-package stage outputs and static graph visualizations."""

from __future__ import annotations

import html
import json
from pathlib import Path

import graphviz
import networkx as nx
import numpy as np
import torch

from mvrfdet.core.constants import MATRIX_SHAPE
from mvrfdet.core.types import MultiViewFeatures
from mvrfdet.pipeline import PipelineArtifacts
from mvrfdet.stage2.feature_matrix import FeatureMatrixGenerator
from mvrfdet.stage3.model import LightweightMVRFDet


MAX_VISUAL_NODES = 160


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _source_label(path: str | Path, source_root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(source_root.resolve()))
    except ValueError:
        return Path(path).name


def _vector_summary(vector: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(vector.shape),
        "nonzero": int(np.count_nonzero(vector)),
        "l2_norm": float(np.linalg.norm(vector)),
        "min": float(vector.min()),
        "max": float(vector.max()),
    }


def _write_svg(dot: graphviz.Digraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".dot").write_text(dot.source, encoding="utf-8")
    path.write_bytes(dot.pipe(format="svg"))


def _draw_api_graph(artifacts: PipelineArtifacts, source_root: Path, path: Path) -> dict[str, object]:
    graph = artifacts.api_graph
    suspicious_ids = {item.node_id for item in artifacts.suspicious}
    selected = list(dict.fromkeys(
        [node_id for node_id in graph.nodes if node_id in suspicious_ids]
        + list(graph.nodes)
    ))[:MAX_VISUAL_NODES]
    selected_set = set(selected)
    dot_ids = {node_id: f"api_{index}" for index, node_id in enumerate(selected)}
    dot = graphviz.Digraph("api_call_graph", graph_attr={"rankdir": "LR", "bgcolor": "white"})
    dot.attr("node", shape="box", style="rounded,filled", fontname="Arial", fontsize="10")
    if not selected:
        dot.node("empty", "No API calls found", fillcolor="#eef2f7")
    for node_id in selected:
        node = graph.nodes[node_id]
        label = f"{node.api_name}\n{_source_label(node.file, source_root)}:{node.line}"
        dot.node(dot_ids[node_id], label, fillcolor="#fed7aa" if node_id in suspicious_ids else "#dbeafe")
    for source, target in graph.edges:
        if source in selected_set and target in selected_set:
            dot.edge(dot_ids[source], dot_ids[target])
    _write_svg(dot, path)
    return {"shown_nodes": len(selected), "total_nodes": len(graph.nodes), "truncated": len(selected) < len(graph.nodes)}


def _draw_cfg(graph: nx.DiGraph, path: Path) -> dict[str, object]:
    selected = list(graph.nodes)[:MAX_VISUAL_NODES]
    selected_set = set(selected)
    dot = graphviz.Digraph("cfg", graph_attr={"rankdir": "TB", "bgcolor": "white"})
    dot.attr("node", shape="box", style="rounded,filled", fontname="Arial", fontsize="10")
    if not selected:
        dot.node("empty", "No valid CFG nodes for this context", fillcolor="#fee2e2")
    for node_id in selected:
        attrs = graph.nodes[node_id]
        label = f"{attrs.get('stmt_type', 'statement')}\nline {attrs.get('line', '?')}"
        color = "#bbf7d0" if attrs.get("entry") else "#fecaca" if attrs.get("terminal") or attrs.get("exit") else "#e2e8f0"
        dot.node(str(node_id), label, fillcolor=color)
    for source, target in graph.edges:
        if source in selected_set and target in selected_set:
            dot.edge(str(source), str(target))
    _write_svg(dot, path)
    return {"shown_nodes": len(selected), "total_nodes": graph.number_of_nodes(), "truncated": len(selected) < graph.number_of_nodes()}


def _report_html(
    package_name: str, diagnostics: dict, cfg_count: int, stage3: dict,
    llm_case: bool = False, stages: set[str] | None = None,
) -> str:
    stages = stages or {"stage1", "stage2", "stage3"}
    title = html.escape(package_name)
    stats = "".join(
        f"<li><b>{html.escape(str(key))}:</b> {html.escape(str(value))}</li>"
        for key, value in diagnostics.items()
        if key in {"static_backend", "api_nodes", "api_edges", "suspicious_occurrences", "unique_contexts", "cfg_count"}
    )
    cfg_links = "".join(
        f'<details><summary>CFG {index + 1} · '
        f'<a href="stage1/cfg/{index:03d}.json">graph data</a></summary>'
        f'<img src="stage1/cfg/{index:03d}.svg" alt="CFG {index + 1}"></details>'
        for index in range(cfg_count)
    ) or "<p>No suspicious API context produced a CFG.</p>"
    stage3_status = html.escape(str(stage3["status"]))
    llm_section = (
        '<section><h2>Stage 0 · LLM API verification</h2>'
        '<p><a href="stage0/llm_reasoning.json">Reasoning case and result</a></p></section>'
        if llm_case else ""
    )
    stage1_section = """<section><h2>Stage 1 · Feature extraction</h2><ul>{stats}</ul>
<p><a href="stage1/metadata.json">Metadata fields and Vm summary</a> ·
<a href="stage1/code.json">Contexts and Vc summary</a> ·
<a href="stage1/graph.json">Graph view and Vg summary</a> ·
<a href="stage1/features.npz">Vm/Vc/Vg arrays</a></p>
<h3>API call graph</h3><p>Orange nodes mark matched sensitive APIs. Large visualizations show up to {MAX_VISUAL_NODES} nodes; the JSON contains the full graph.</p>
<p><a href="stage1/api_call_graph.json">Full graph data</a> ·
<a href="stage1/api_call_graph.svg">SVG</a> · <a href="stage1/api_call_graph.dot">DOT</a></p>
<img src="stage1/api_call_graph.svg" alt="API call graph">
<h3>Control flow graphs</h3><p>Green marks an entry; red marks an exit or terminal node.</p>{cfg_links}</section>""".format(
        stats=stats, MAX_VISUAL_NODES=MAX_VISUAL_NODES, cfg_links=cfg_links
    ) if "stage1" in stages else ""
    stage2_section = (
        '<section><h2>Stage 2 · Feature matrix</h2><p><a href="stage2/matrix.json">Shape and checks</a> · '
        '<a href="stage2/matrix.npy">Matrix [3,32,32]</a></p></section>'
        if "stage2" in stages else ""
    )
    stage3_section = (
        f'<section><h2>Stage 3 · Classification</h2><p>Status: <code>{html.escape(str(stage3["status"]))}</code>. '
        '<a href="stage3/inference.json">Inference details</a></p></section>'
        if "stage3" in stages else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title} · MVRFDet report</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#172033}}
h1,h2{{line-height:1.2}}section{{border:1px solid #d8e0eb;border-radius:12px;padding:1rem 1.3rem;margin:1.2rem 0}}
img{{max-width:100%;max-height:780px;border:1px solid #e2e8f0}}a{{color:#0953b8}}code{{background:#eef2f7;padding:.1rem .3rem}}</style>
</head><body><h1>{title}</h1><p>Static analysis report. Graphs show the lightweight AST backend when selected.</p>
{llm_section}
{stage1_section}
{stage2_section}
{stage3_section}
</body></html>"""


def write_feature_matrix(output_dir: Path, features: MultiViewFeatures) -> np.ndarray:
    matrix = FeatureMatrixGenerator().generate_numpy(features)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "matrix.npy", matrix)
    _write_json(output_dir / "matrix.json", {
        "shape": list(matrix.shape), "channels": ["metadata", "code", "graph"],
        "channel_l2_norms": [float(np.linalg.norm(channel)) for channel in matrix],
        "finite": bool(np.isfinite(matrix).all()),
    })
    return matrix


def load_classifier(checkpoint: str | Path) -> LightweightMVRFDet:
    model = LightweightMVRFDet().eval()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state"] if "model_state" in saved else saved)
    return model


def write_inference(
    output_dir: Path, matrix: np.ndarray, checkpoint: str | Path, model: LightweightMVRFDet,
) -> dict[str, object]:
    if not isinstance(matrix, np.ndarray) or matrix.shape != MATRIX_SHAPE or matrix.dtype.kind not in "fiu":
        raise ValueError(f"Expected a numeric matrix of shape {MATRIX_SHAPE}")
    matrix = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains NaN or infinity")
    with torch.no_grad():
        logits, attention = model(torch.from_numpy(matrix).unsqueeze(0), return_attention=True)
        probabilities = torch.softmax(logits, dim=1)[0].tolist()
    result = {
        "status": "completed", "checkpoint": str(checkpoint),
        "prediction": "malicious" if probabilities[1] >= probabilities[0] else "benign",
        "probabilities": {"benign": probabilities[0], "malicious": probabilities[1]},
        "attention_mean": float(attention.mean()),
    }
    _write_json(output_dir / "inference.json", result)
    return result


def write_package_report(
    output_dir: str | Path,
    package_name: str,
    source_root: str | Path,
    artifacts: PipelineArtifacts,
    checkpoint: str | Path | None = None,
    stages: set[str] | None = None,
    *,
    classifier: LightweightMVRFDet | None = None,
) -> dict[str, object]:
    """Save vectors, intermediate graph data, SVGs, and a browsable report."""
    root = Path(output_dir)
    stages = stages or {"stage1", "stage2", "stage3"}
    unknown_stages = stages - {"stage1", "stage2", "stage3"}
    if unknown_stages:
        raise ValueError(f"Unknown report stages: {sorted(unknown_stages)}")
    source_root = Path(source_root)
    stage1, stage2, stage3 = root / "stage1", root / "stage2", root / "stage3"
    for stage_name, stage_path in (("stage1", stage1), ("stage2", stage2), ("stage3", stage3)):
        if stage_name in stages:
            stage_path.mkdir(parents=True, exist_ok=True)

    features = artifacts.features
    diagnostics = dict(features.diagnostics)
    diagnostics["package"] = package_name
    diagnostics["pkg_info"] = _source_label(diagnostics["pkg_info"], source_root)

    if "stage1" in stages:
        features.save_npz(stage1 / "features.npz")
        _write_json(stage1 / "metadata.json", {
            "source": diagnostics["pkg_info"], "fields": artifacts.metadata,
            "vm": _vector_summary(features.metadata),
        })
        _write_json(stage1 / "code.json", {
            "suspicious_occurrences": [
                {"api_name": item.api_name, "file": _source_label(item.file, source_root),
                 "line": item.line, "column": item.column, "node_id": item.node_id}
                for item in artifacts.suspicious
            ],
            "contexts": [
                {"api_name": item.api_name, "file": _source_label(item.source_file, source_root),
                 "center_line": item.center_line, "token_count": item.token_count,
                 "digest": item.digest, "text": item.text}
                for item in artifacts.contexts
            ],
            "vc": _vector_summary(features.code),
        })

        graph = artifacts.api_graph
        _write_json(stage1 / "api_call_graph.json", {
            "backend": graph.backend,
            "nodes": [
                {"id": node.id, "api_name": node.api_name, "file": _source_label(node.file, source_root),
                 "line": node.line, "column": node.column, "function": node.function,
                 "suspicious": node.id in {item.node_id for item in artifacts.suspicious}}
                for node in graph.nodes.values()
            ],
            "edges": [{"source": a, "target": b} for a, b in graph.edges],
        })
        api_visual = _draw_api_graph(artifacts, source_root, stage1 / "api_call_graph.svg")
        cfg_visuals = []
        for index, cfg in enumerate(artifacts.cfgs):
            cfg_path = stage1 / "cfg" / f"{index:03d}"
            _write_json(cfg_path.with_suffix(".json"), {
                "context_index": index,
                "nodes": [{"id": str(node), **attrs} for node, attrs in cfg.nodes(data=True)],
                "edges": [{"source": str(a), "target": str(b)} for a, b in cfg.edges()],
            })
            cfg_visuals.append(_draw_cfg(cfg, cfg_path.with_suffix(".svg")))
        _write_json(stage1 / "graph.json", {
            "api_call_graph": {"nodes": len(graph.nodes), "edges": len(graph.edges), "visualization": api_visual},
            "cfgs": [
                {"index": index, "nodes": cfg.number_of_nodes(), "edges": cfg.number_of_edges(),
                 "visualization": cfg_visuals[index]}
                for index, cfg in enumerate(artifacts.cfgs)
            ],
            "vg": _vector_summary(features.graph),
        })

    matrix = write_feature_matrix(stage2, features) if "stage2" in stages else None
    if "stage3" not in stages:
        inference = {"status": "not_run_stage_not_selected"}
    elif checkpoint is None:
        inference = {"status": "not_run_no_checkpoint"}
        _write_json(stage3 / "inference.json", inference)
    else:
        if matrix is None:
            matrix = FeatureMatrixGenerator().generate_numpy(features)
        if classifier is None:
            classifier = load_classifier(checkpoint)
        inference = write_inference(stage3, matrix, checkpoint, classifier)
    stage_status = {name: {"status": "completed" if name in stages else "not_run"} for name in ("stage1", "stage2")}
    stage_status["stage3"] = inference
    _write_json(root / "summary.json", {"package": package_name, **diagnostics,
                                        "stages": stage_status, "stage3": inference, "report": "report.html"})
    has_llm_case = (root / "stage0" / "llm_reasoning.json").exists()
    (root / "report.html").write_text(
        _report_html(package_name, diagnostics, len(artifacts.cfgs), inference,
                     llm_case=has_llm_case, stages=stages),
        encoding="utf-8",
    )
    return {"package": package_name, **diagnostics, "stages": stage_status,
            "stage3_status": inference["status"], "report": str(root / "report.html")}


def save_llm_case(output_dir: str | Path, result: dict) -> Path:
    """Attach LLM verification results to an existing package report."""
    root = Path(output_dir)
    path = root / "stage0" / "llm_reasoning.json"
    _write_json(path, result)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    summary["llm_status"] = result.get("status")
    _write_json(root / "summary.json", summary)
    report = _report_html(summary["package"], summary, summary["cfg_count"], summary["stage3"], llm_case=True)
    (root / "report.html").write_text(report, encoding="utf-8")
    return path
