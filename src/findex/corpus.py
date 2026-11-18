import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Lazily yield Document objects from .txt/.md files or .jsonl files under root."""
    if not root.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root}")

    paths = [root] if root.is_file() else sorted(root.rglob("*"))

    for path in paths:
        if not path.is_file():
            continue

        if path.suffix == ".jsonl":
            yield from _iter_jsonl(path)
        elif path.suffix in {".txt", ".md"}:
            doc = _read_text_file(path, root)
            if doc is not None:
                yield doc


def _read_text_file(path: Path, root: Path) -> Document | None:
    try:
        raw_bytes = path.read_bytes()
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("Invalid UTF-8 in %s; decoding with replacement.", path)
            text = raw_bytes.decode("utf-8", errors="replace")

        doc_id = str(path.relative_to(root)) if root.is_dir() else path.name
        return Document(doc_id=doc_id, path=path, text=text)
    except OSError as exc:
        logger.error("Failed to read %s: %s", path, exc)
        return None


def _iter_jsonl(path: Path) -> Iterator[Document]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                text = record.get("text") or record.get("abstract") or ""
                doc_id = str(record.get("id", f"{path.name}:{line_no}"))
                yield Document(doc_id=doc_id, path=path, text=text)
            except json.JSONDecodeError:
                logger.warning("Malformed JSON at %s:%d; skipping.", path, line_no)
