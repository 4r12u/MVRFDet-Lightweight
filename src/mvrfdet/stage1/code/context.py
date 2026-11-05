from __future__ import annotations

import hashlib
import io
import tokenize
from pathlib import Path

from mvrfdet.core.constants import MAX_CONTEXT_TOKENS
from mvrfdet.core.types import CodeContextSnippet, SuspiciousAPIOccurrence


def _tokenize_source(source: str) -> list[tokenize.TokenInfo]:
    try:
        return list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError):
        return []


def extract_code_context(
    occurrence: SuspiciousAPIOccurrence,
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> CodeContextSnippet:
    source = occurrence.file.read_text(encoding="utf-8", errors="replace")
    tokens = _tokenize_source(source)
    if not tokens:
        # Syntax-broken source fallback: use a centered line window. The token
        # count is approximate in this branch.
        lines = source.splitlines()
        center = max(0, min(len(lines) - 1, occurrence.line - 1)) if lines else 0
        half = max(1, max_tokens // 8)
        start = max(0, center - half)
        end = min(len(lines), center + half + 1)
        text = "\n".join(lines[start:end])
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        return CodeContextSnippet(occurrence.api_name, occurrence.file, occurrence.line, text, min(max_tokens, len(text.split())), digest)

    # Exclude ENCODING/ENDMARKER for window accounting when possible.
    useful = [tok for tok in tokens if tok.type not in {tokenize.ENCODING, tokenize.ENDMARKER}]
    center_idx = 0
    best_distance = None
    for idx, tok in enumerate(useful):
        distance = abs(tok.start[0] - occurrence.line) * 10000 + abs(tok.start[1] - occurrence.column)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            center_idx = idx

    budget = max(1, min(max_tokens, len(useful)))
    left_budget = budget // 2
    right_budget = budget - left_budget
    start = max(0, center_idx - left_budget)
    end = min(len(useful), center_idx + right_budget)
    # Redistribute unused budget when one side reaches the file boundary.
    used = end - start
    if used < budget:
        missing = budget - used
        shift_left = min(start, missing)
        start -= shift_left
        missing -= shift_left
        end = min(len(useful), end + missing)

    selected = useful[start:end]
    try:
        text = tokenize.untokenize(selected)
    except Exception:
        # Untokenize may reject an isolated INDENT/DEDENT boundary; line slice
        # remains a robust fallback while preserving the centered intent.
        min_line = min((tok.start[0] for tok in selected), default=occurrence.line)
        max_line = max((tok.end[0] for tok in selected), default=occurrence.line)
        lines = source.splitlines()
        text = "\n".join(lines[max(0, min_line - 1):max_line])
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return CodeContextSnippet(
        api_name=occurrence.api_name,
        source_file=occurrence.file,
        center_line=occurrence.line,
        text=text,
        token_count=len(selected),
        digest=digest,
    )


def extract_contexts(
    occurrences: list[SuspiciousAPIOccurrence],
    max_tokens: int = MAX_CONTEXT_TOKENS,
    deduplicate_identical: bool = True,
) -> list[CodeContextSnippet]:
    snippets = [extract_code_context(item, max_tokens=max_tokens) for item in occurrences]
    if not deduplicate_identical:
        return snippets
    seen: set[str] = set()
    result: list[CodeContextSnippet] = []
    for snippet in snippets:
        if snippet.digest in seen:
            continue
        seen.add(snippet.digest)
        result.append(snippet)
    return result
