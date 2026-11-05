from pathlib import Path
import json

from mvrfdet.offline.sensitive_api.deepseek import DeepSeekVerifier, load_paper_prompt, render_prompt


def test_prompt_rendering():
    path = Path("prompts/sensitive_api/paper_prompt.txt")
    template = load_paper_prompt(path)
    prompt = render_prompt(template, "os.system", ["os.system('id')"])
    assert "Candidate API: os.system" in prompt
    assert "os.system('id')" in prompt
    assert "INSERT<Suspicious APIs Code Context Snippets>" not in prompt


def test_reasoning_response_is_preserved(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({
                "choices": [{"message": {
                    "content": '{"api_name":"os.system","is_potentially_malicious":true}',
                    "reasoning_content": "Reasoning from the API",
                }, "finish_reason": "stop"}],
                "usage": {"total_tokens": 42},
            }).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    verifier = DeepSeekVerifier(api_key="test-key", model="deepseek-v4-pro")
    result = verifier.verify("Return JSON", thinking=True)

    assert captured["thinking"] == {"type": "enabled"}
    assert "temperature" not in captured
    assert result["reasoning_content"] == "Reasoning from the API"
    assert result["parsed"]["api_name"] == "os.system"
