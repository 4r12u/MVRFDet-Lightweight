from pathlib import Path

from mvrfdet.offline.sensitive_api.builder import verify_candidates


def test_offline_llm_cache_tracks_prompt_and_thinking(tmp_path: Path):
    calls = []

    class FakeVerifier:
        model = "test-model"
        api_url = "https://example.test/chat/completions"

        def verify(self, prompt, retries, thinking, max_tokens):
            calls.append((prompt, thinking, max_tokens))
            return {
                "model": self.model,
                "raw_response": "{}",
                "parsed": {"api_name": "os.system", "is_potentially_malicious": True},
            }

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Code: INSERT<Suspicious APIs Code Context Snippets>", encoding="utf-8")
    candidates = [{"api_name": "os.system", "score": 1}]
    contexts = {"os.system": ["os.system('id')"]}
    kwargs = dict(
        candidates=candidates, contexts=contexts, prompt_path=prompt_path,
        cache_dir=tmp_path / "cache", verifier=FakeVerifier(), thinking=True,
    )

    assert len(verify_candidates(**kwargs)) == 1
    assert len(verify_candidates(**kwargs)) == 1
    assert len(calls) == 1

    contexts["os.system"] = ["os.system('whoami')"]
    assert len(verify_candidates(**kwargs)) == 1
    assert len(calls) == 2
