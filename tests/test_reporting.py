import json
from pathlib import Path

import numpy as np

from mvrfdet.pipeline import LightweightFeaturePipeline
from mvrfdet.reporting import write_package_report


def test_package_report_keeps_stage_outputs_and_rendered_graphs(tmp_path: Path):
    package = tmp_path / "sample"
    package.mkdir()
    (package / "PKG-INFO").write_text("Name: sample\nSummary: report test\n", encoding="utf-8")
    (package / "sample.py").write_text(
        "import os\n"
        "def run(flag):\n"
        "    if flag:\n"
        "        os.system('echo test')\n",
        encoding="utf-8",
    )
    artifacts = LightweightFeaturePipeline(static_backend="ast").extract_with_artifacts(
        package, {"os.system"}
    )
    output = tmp_path / "report"
    result = write_package_report(output, "sample", package, artifacts)

    assert result["stage3_status"] == "not_run_no_checkpoint"
    assert (output / "report.html").exists()
    assert (output / "stage1/api_call_graph.svg").read_text(encoding="utf-8").find("<svg") >= 0
    assert (output / "stage1/cfg/000.svg").read_text(encoding="utf-8").find("<svg") >= 0
    graph = json.loads((output / "stage1/api_call_graph.json").read_text(encoding="utf-8"))
    assert any(node["api_name"] == "os.system" and node["suspicious"] for node in graph["nodes"])
    assert np.load(output / "stage2/matrix.npy").shape == (3, 32, 32)
    assert json.loads((output / "stage3/inference.json").read_text(encoding="utf-8"))["status"] == "not_run_no_checkpoint"
