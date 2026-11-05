from pathlib import Path

from mvrfdet.stage1.metadata.parser import parse_pkg_info
from mvrfdet.stage1.metadata.encoder import StaticMetadataEncoder


def test_metadata_parser_and_encoder(tmp_path: Path):
    pkg = tmp_path / "PKG-INFO"
    pkg.write_text(
        "Metadata-Version: 2.1\n"
        "Name: demo-package\n"
        "Summary: demo summary\n"
        "Home-page: https://github.com/example/demo\n"
        "Author: Alice\n"
        "Author-Email: alice@example.com\n"
        "License: MIT\n"
        "Description: short description\n",
        encoding="utf-8",
    )
    metadata = parse_pkg_info(pkg)
    assert metadata["Name"] == "demo-package"
    assert metadata["License"] == "MIT"
    vector = StaticMetadataEncoder().encode(metadata)
    assert vector.shape == (768,)
