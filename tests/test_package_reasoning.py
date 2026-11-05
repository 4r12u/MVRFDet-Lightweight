import argparse
import json
from pathlib import Path

from mvrfdet.offline.sensitive_api import package_reasoning
from mvrfdet.pipeline import LightweightFeaturePipeline
from mvrfdet.reporting import write_package_report


def test_package_reasoning_writes_stage0_and_reuses_matching_case(tmp_path, monkeypatch):
    package = tmp_path / "sample"
    package.mkdir()
    (package / "PKG-INFO").write_text("Name: sample\n", encoding="utf-8")
    (package / "sample.py").write_text(
        "import subprocess\nsubprocess.Popen(['echo', 'hello'])\n", encoding="utf-8"
    )
    sensitive = {"subprocess.Popen"}
    artifacts = LightweightFeaturePipeline(static_backend="ast").extract_with_artifacts(package, sensitive)
    report_dir = tmp_path / "report"
    write_package_report(report_dir, "sample", package, artifacts)

    calls = []

    class FakeVerifier:
        def __init__(self, api_key, api_url, model, timeout):
            assert api_key == "dummy-key"
            self.api_url = api_url or "https://api.deepseek.com/chat/completions"

        def verify(self, prompt, retries, thinking, max_tokens):
            assert "subprocess.Popen" in prompt
            assert "echo" in prompt
            assert thinking is True
            calls.append(prompt)
            return {"parsed": {"api_name": "subprocess.Popen", "is_potentially_malicious": True}}

    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "dummy-key")
    monkeypatch.setattr(package_reasoning, "DeepSeekVerifier", FakeVerifier)
    parser = argparse.ArgumentParser()
    package_reasoning.add_package_reasoning_arguments(parser)
    args = parser.parse_args(["--llm-api-key-env", "TEST_DEEPSEEK_KEY", "--llm-api", "subprocess.Popen"])
    options = package_reasoning.options_from_args(args)

    first = package_reasoning.run_package_reasoning(artifacts, sensitive, report_dir, options)
    second = package_reasoning.run_package_reasoning(artifacts, sensitive, report_dir, options)

    assert first["llm_status"] == "completed"
    assert first["llm_cached"] == 0
    assert second["llm_cached"] == 1
    assert len(calls) == 1
    saved = json.loads((report_dir / "stage0/llm_reasoning.json").read_text(encoding="utf-8"))
    assert saved["cases"][0]["parsed"]["api_name"] == "subprocess.Popen"
    assert "dummy-key" not in (report_dir / "stage0/llm_reasoning.json").read_text(encoding="utf-8")
    assert "stage0/llm_reasoning.json" in (report_dir / "report.html").read_text(encoding="utf-8")


def test_package_reasoning_requires_explicit_key_env(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    parser = argparse.ArgumentParser()
    package_reasoning.add_package_reasoning_arguments(parser)
    args = parser.parse_args([])
    try:
        package_reasoning.options_from_args(args)
    except package_reasoning.DeepSeekVerificationError as exc:
        assert "DEEPSEEK_API_KEY" in str(exc)
    else:
        raise AssertionError("Missing key should fail before a request")
