from __future__ import annotations

import math
import re
from urllib.parse import urlparse

import numpy as np

from mvrfdet.core.constants import METADATA_FIELDS


_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_URL_RE = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
_KNOWN_LICENSE_WORDS = {
    "mit", "apache", "bsd", "gpl", "lgpl", "mpl", "isc", "unlicense", "proprietary",
}


def _safe(value: str | None) -> str:
    return (value or "").strip()


def _scaled_length(value: str, scale: float = 1000.0) -> float:
    return min(1.0, math.log1p(len(value)) / math.log1p(scale))


def _word_count(value: str, scale: float = 300.0) -> float:
    return min(1.0, len(re.findall(r"\b\w+\b", value)) / scale)


def _ratio(value: str, predicate) -> float:
    if not value:
        return 0.0
    return sum(1 for char in value if predicate(char)) / len(value)


def _valid_url(value: str) -> float:
    if not value:
        return 0.0
    try:
        parsed = urlparse(value)
        return float(parsed.scheme in {"http", "https"} and bool(parsed.netloc))
    except Exception:
        return 0.0


def handcrafted_metadata_features(metadata: dict[str, str]) -> np.ndarray:
    """Return exactly 32 cheap, deterministic metadata features."""
    m = {field: _safe(metadata.get(field, "")) for field in METADATA_FIELDS}
    name, summary, homepage = m["Name"], m["Summary"], m["Home-page"]
    author, email, license_, description = m["Author"], m["Author-Email"], m["License"], m["Description"]
    desc_words = re.findall(r"\b\w+\b", description.lower())
    unique_ratio = len(set(desc_words)) / len(desc_words) if desc_words else 0.0
    completeness = sum(bool(m[field]) for field in METADATA_FIELDS) / len(METADATA_FIELDS)
    features = [
        float(bool(name)), float(bool(summary)), float(bool(homepage)), float(bool(author)),
        float(bool(email)), float(bool(license_)), float(bool(description)),
        _scaled_length(name, 100), _scaled_length(summary, 1000), _scaled_length(homepage, 500),
        _scaled_length(author, 300), _scaled_length(email, 300), _scaled_length(license_, 300),
        _scaled_length(description, 10000), _word_count(name, 30), _word_count(summary, 100),
        _word_count(description, 1000), unique_ratio,
        _ratio(description, str.isdigit), _ratio(description, lambda c: not c.isalnum() and not c.isspace()),
        _ratio(name, str.isdigit), _ratio(name, lambda c: c in "-_"),
        float(bool(_EMAIL_RE.match(email))), _valid_url(homepage),
        float("github.com" in homepage.lower()), float(homepage.lower().startswith(("http://", "https://"))),
        float(any(word in license_.lower() for word in _KNOWN_LICENSE_WORDS)),
        float(bool(_URL_RE.search(description))),
        float(any(marker in description for marker in ("```", "::", ">>>"))),
        float(not summary or len(summary) < 8), float(not description or len(description) < 20),
        completeness,
    ]
    return np.asarray(features, dtype=np.float32)


def metadata_text(metadata: dict[str, str]) -> str:
    return "\n".join(f"{field}: {_safe(metadata.get(field, ''))}" for field in METADATA_FIELDS)
