from pathlib import Path

from mvrfdet.pipeline import LightweightFeaturePipeline
from mvrfdet.stage2.feature_matrix import FeatureMatrixGenerator


def test_end_to_end_feature_pipeline(tmp_path: Path):
    package = tmp_path / "demo"
    package.mkdir()
    (package / "PKG-INFO").write_text(
        "Metadata-Version: 2.1\n"
        "Name: demo\n"
        "Summary: demo\n"
        "Home-page: https://example.com\n"
        "Author: A\n"
        "Author-Email: a@example.com\n"
        "License: MIT\n"
        "Description: demo package\n",
        encoding="utf-8",
    )
    (package / "demo.py").write_text(
        "import os\n"
        "import socket\n\n"
        "def go():\n"
        "    s = socket.socket()\n"
        "    os.system('id')\n"
        "    return s\n",
        encoding="utf-8",
    )
    features = LightweightFeaturePipeline(static_backend="ast").extract(
        package,
        {"os.system", "socket.socket"},
    )
    assert features.metadata.shape == (768,)
    assert features.code.shape == (768,)
    assert features.graph.shape == (1024,)
    assert features.diagnostics["suspicious_occurrences"] >= 2
    matrix = FeatureMatrixGenerator().generate(features)
    assert tuple(matrix.shape) == (3, 32, 32)
