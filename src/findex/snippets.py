from __future__ import annotations

import re
import unicodedata
from pathlib import Path


def build_snippet(
    doc_path: str | Path,
    terms: list[str] | tuple[str, ...],
    *,
    window: int = 80,
) -> str:
    """Extract a +-window character snippet around the best match and highlight matched terms."""
    path = Path(doc_path)
    if not path.exists():
        return ""

    raw_text = path.read_text(encoding="utf-8", errors="replace")
    clean_text = " ".join(raw_text.split())
    if not clean_text:
        return ""

    unique_terms = [t for t in dict.fromkeys(terms) if t]
    if not unique_terms:
        return clean_text[: window * 2] + ("..." if len(clean_text) > window * 2 else "")

    folded_text = unicodedata.normalize("NFC", clean_text).casefold()
    best_pos = -1
    for term in unique_terms:
        idx = folded_text.find(term)
        if idx != -1 and (best_pos == -1 or idx < best_pos):
            best_pos = idx

    if best_pos == -1:
        start = 0
        end = min(len(clean_text), window * 2)
    else:
        start = max(0, best_pos - window)
        end = min(len(clean_text), best_pos + window)

    snippet = clean_text[start:end]
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(clean_text) else ""

    pattern = re.compile(
        r"\b("
        + "|".join(re.escape(t) for t in sorted(unique_terms, key=len, reverse=True))
        + r")\b",
        flags=re.IGNORECASE | re.UNICODE,
    )
    highlighted = pattern.sub(r"[\1]", snippet)
    return f"{prefix}{highlighted}{suffix}"
