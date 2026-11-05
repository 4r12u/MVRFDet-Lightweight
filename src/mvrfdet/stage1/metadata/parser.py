from __future__ import annotations

import re
from pathlib import Path

from mvrfdet.core.constants import METADATA_FIELDS


_HEADER_RE = re.compile(r"^([A-Za-z][A-Za-z0-9-]*):[ \t]*(.*)$")


def locate_pkg_info(package_path: str | Path) -> Path:
    """Locate the most likely PKG-INFO/METADATA file below a package root."""
    root = Path(package_path)
    if root.is_file() and root.name in {"PKG-INFO", "METADATA"}:
        return root
    candidates = list(root.rglob("PKG-INFO")) + list(root.rglob("METADATA"))
    if not candidates:
        raise FileNotFoundError(f"No PKG-INFO or METADATA found under {root}")
    candidates.sort(key=lambda p: (0 if p.name == "PKG-INFO" else 1, len(p.parts), str(p)))
    return candidates[0]


def parse_pkg_info(path: str | Path) -> dict[str, str]:
    """Parse the seven paper-selected metadata fields using header regexes.

    Continuation lines are appended to the preceding header. If Description is
    absent as a header, the body after the first blank line is used as a
    fallback because many PKG-INFO files store the long description there.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    values: dict[str, list[str]] = {field: [] for field in METADATA_FIELDS}
    current: str | None = None
    body_start: int | None = None

    for index, line in enumerate(lines):
        if body_start is None and line.strip() == "":
            body_start = index + 1
        match = _HEADER_RE.match(line)
        if match:
            key, value = match.group(1), match.group(2)
            current = key if key in values else None
            if current:
                values[current].append(value.strip())
            continue
        if current and (line.startswith(" ") or line.startswith("\t")):
            values[current].append(line.strip())

    result = {field: "\n".join(part for part in values[field] if part).strip() for field in METADATA_FIELDS}
    if not result["Description"] and body_start is not None:
        result["Description"] = "\n".join(lines[body_start:]).strip()
    return result
