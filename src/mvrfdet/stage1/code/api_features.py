from __future__ import annotations

import ast
from collections import Counter

import numpy as np
from sklearn.feature_extraction import FeatureHasher


CATEGORY_PREFIXES: dict[str, tuple[str, ...]] = {
    "network": ("socket", "requests", "urllib", "http", "ftplib", "smtplib", "paramiko"),
    "filesystem": ("open", "pathlib", "os.remove", "os.unlink", "os.rename", "shutil", "tempfile"),
    "process": ("subprocess", "multiprocessing", "os.fork", "os.kill", "pty"),
    "shell": ("os.system", "subprocess.Popen", "subprocess.run", "subprocess.call", "pty.spawn"),
    "dynamic_execution": ("eval", "exec", "compile", "importlib", "__import__"),
    "encoding": ("base64", "binascii", "codecs"),
    "serialization": ("pickle", "marshal", "shelve", "yaml.load"),
    "environment": ("os.environ", "os.getenv", "platform", "sys.platform"),
    "credentials": ("keyring", "getpass", "netrc"),
    "system_information": ("platform", "socket.gethostname", "os.getlogin", "psutil"),
}


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _qualified_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def extract_api_calls(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    calls: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            calls.append((int(getattr(node, "lineno", 0)), int(getattr(node, "col_offset", 0)), _qualified_name(node.func) or "<dynamic-call>"))
    calls.sort()
    return [name for _, _, name in calls]


def api_feature_dict(source: str) -> dict[str, float]:
    calls = extract_api_calls(source)
    counts = Counter(calls)
    result: dict[str, float] = {f"api:{name}": float(count) for name, count in counts.items()}
    category_counts = Counter()
    for name in calls:
        lowered = name.lower()
        for category, prefixes in CATEGORY_PREFIXES.items():
            if any(lowered == prefix.lower() or lowered.startswith(prefix.lower() + ".") for prefix in prefixes):
                category_counts[category] += 1
    result.update({f"category:{name}": float(count) for name, count in category_counts.items()})
    result["api_total"] = float(len(calls))
    result["api_unique"] = float(len(counts))
    result["api_repetition_ratio"] = float(1.0 - len(counts) / len(calls)) if calls else 0.0
    return result


class APIFeatureEncoder:
    def __init__(self, dimension: int = 256):
        self.dimension = dimension
        self.hasher = FeatureHasher(n_features=dimension, input_type="dict", alternate_sign=False)

    def encode(self, source: str) -> np.ndarray:
        return self.hasher.transform([api_feature_dict(source)]).toarray()[0].astype(np.float32)
