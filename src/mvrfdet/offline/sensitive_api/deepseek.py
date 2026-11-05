from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


class DeepSeekVerificationError(RuntimeError):
    pass


def load_paper_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def render_prompt(template: str, api_name: str, contexts: list[str]) -> str:
    body_parts = [f"Candidate API: {api_name}"]
    for index, context in enumerate(contexts, 1):
        body_parts.append(f"\n--- Context {index} ---\n{context}")
    body = "\n".join(body_parts)
    return template.replace("INSERT<Suspicious APIs Code Context Snippets>", body)


def _parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise DeepSeekVerificationError("LLM response did not contain a JSON object")
        return json.loads(match.group(0))


class DeepSeekVerifier:
    """Minimal OpenAI-compatible DeepSeek Chat Completions client.

    Endpoint/model are environment-configurable so the project does not guess
    the account-specific identifier for the user's DeepSeek V4 Pro access.
    """

    def __init__(self, api_key: str | None = None, api_url: str | None = None, model: str | None = None, timeout: int = 90):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.api_url = api_url or os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
        self.model = model or os.environ.get("DEEPSEEK_MODEL")
        self.timeout = timeout
        if not self.api_key:
            raise DeepSeekVerificationError("DEEPSEEK_API_KEY is not set")
        if not self.model:
            raise DeepSeekVerificationError("DEEPSEEK_MODEL is not set; use the exact model id available in your account")

    def verify(self, prompt: str, retries: int = 3, thinking: bool = False, max_tokens: int = 4096) -> dict:
        if retries < 1 or max_tokens < 1:
            raise ValueError("retries and max_tokens must be positive")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": max_tokens,
        }
        if thinking:
            payload["thinking"] = {"type": "enabled"}
        else:
            payload["temperature"] = 0
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                choice = data["choices"][0]
                message = choice["message"]
                content = message["content"]
                if not isinstance(content, str):
                    raise DeepSeekVerificationError("LLM response content is not text")
                parsed = _parse_json_response(content)
                return {
                    "raw_response": content,
                    "parsed": parsed,
                    "model": self.model,
                    "reasoning_content": message.get("reasoning_content"),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": data.get("usage"),
                }
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError, DeepSeekVerificationError) as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(2 ** attempt)
        raise DeepSeekVerificationError(f"DeepSeek verification failed after {retries} attempts: {last_error}")
