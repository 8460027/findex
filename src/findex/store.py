from __future__ import annotations

import argparse
import json
import pickle
import struct
import sys
import time
from array import array
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, Literal, cast

from findex.decorators import timed
from findex.index import Index, build_index
from findex.models import DocMeta, Posting

FormatType = Literal["pickle", "json", "binary"]

MAGIC: Final[bytes] = b"FIDX0002"
HEADER_STRUCT: Final[struct.Struct] = struct.Struct("<8sI")


def save_index(index: Index, path: str | Path, fmt: FormatType = "binary") -> None:
    """Persist an Index to disk using pickle, json, or custom binary format."""
    target = Path(path)

    if fmt == "pickle":
        with target.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    if fmt == "json":
        payload: dict[str, Any] = {
            "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
            "doc_meta": {
                str(k): {"title": m.title, "path": m.path} for k, m in index.doc_meta.items()
            },
            "postings": {
                term: [[p.doc_id, p.tf, list(p.positions)] for p in plist]
                for term, plist in index.postings.items()
            },
        }
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return

    all_doc_ids = array("I")
    all_tfs = array("I")
    all_pos_counts = array("I")
    all_positions = array("I")

    term_table: dict[str, tuple[int, int]] = {}
    offset = 0
    for term, plist in index.postings.items():
        length = len(plist)
        term_table[term] = (offset, length)
        for p in plist:
            all_doc_ids.append(p.doc_id)
            all_tfs.append(p.tf)
            all_pos_counts.append(len(p.positions))
            if p.positions:
                all_positions.extend(p.positions)
        offset += length

    header_obj: dict[str, Any] = {
        "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
        "doc_meta": {str(k): [m.title, m.path] for k, m in index.doc_meta.items()},
        "terms": term_table,
        "total_postings": offset,
        "total_positions": len(all_positions),
    }
    header_bytes = json.dumps(header_obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    with target.open("wb") as f:
        f.write(HEADER_STRUCT.pack(MAGIC, len(header_bytes)))
        f.write(header_bytes)
        all_doc_ids.tofile(f)
        all_tfs.tofile(f)
        all_pos_counts.tofile(f)
        all_positions.tofile(f)


@timed
def load_index(path: str | Path, fmt: FormatType = "binary") -> Index:
    """Load an Index from disk."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Index file not found: {source}")

    if fmt == "pickle":
        # SECURITY NOTE: pickle.load executes arbitrary opcodes; never load untrusted files.
        with source.open("rb") as f:
            loaded: object = pickle.load(f)
        if isinstance(loaded, Index):
            return loaded
        raise TypeError(f"Expected Index in pickle file, got {type(loaded).__name__}")

    if fmt == "json":
        raw = cast(dict[str, Any], json.loads(source.read_text(encoding="utf-8")))
        raw_lengths = cast(dict[str, int], raw["doc_lengths"])
        raw_meta = cast(dict[str, dict[str, str]], raw["doc_meta"])
        raw_postings = cast(dict[str, list[list[Any]]], raw["postings"])

        doc_lengths = {int(k): int(v) for k, v in raw_lengths.items()}
        doc_meta = {
            int(k): DocMeta(doc_id=int(k), title=v["title"], path=v["path"])
            for k, v in raw_meta.items()
        }
        postings = {
            term: [
                Posting(
                    doc_id=int(item[0]),
                    tf=int(item[1]),
                    positions=tuple(int(pos) for pos in cast(list[int], item[2])),
                )
                for item in plist
            ]
            for term, plist in raw_postings.items()
        }
        return Index(postings=postings, doc_lengths=doc_lengths, doc_meta=doc_meta)

    with source.open("rb") as f:
        prefix = f.read(HEADER_STRUCT.size)
        if len(prefix) != HEADER_STRUCT.size:
            raise ValueError(f"Corrupted or truncated index file: {source}")
        unpacked = cast(tuple[bytes, int], HEADER_STRUCT.unpack(prefix))
        magic, header_len = unpacked
        if magic != MAGIC:
            raise ValueError(f"Invalid index file magic: {magic!r}")
        header_obj = cast(dict[str, Any], json.loads(f.read(header_len).decode("utf-8")))

        total_postings = int(header_obj["total_postings"])
        total_positions = int(header_obj["total_positions"])

        all_doc_ids = array("I")
        all_doc_ids.fromfile(f, total_postings)

        all_tfs = array("I")
        all_tfs.fromfile(f, total_postings)

        all_pos_counts = array("I")
        all_pos_counts.fromfile(f, total_postings)

        all_positions = array("I")
        if total_positions > 0:
            all_positions.fromfile(f, total_positions)

    raw_doc_lengths = cast(dict[str, int], header_obj["doc_lengths"])
    raw_doc_meta = cast(dict[str, list[str]], header_obj["doc_meta"])
    raw_terms = cast(dict[str, list[int]], header_obj["terms"])

    doc_lengths = {int(k): int(v) for k, v in raw_doc_lengths.items()}
    doc_meta = {
        int(k): DocMeta(doc_id=int(k), title=v[0], path=v[1]) for k, v in raw_doc_meta.items()
    }

    postings_map: dict[str, list[Posting]] = {}
    pos_cursor = 0
    for term, span in raw_terms.items():
        offset, length = int(span[0]), int(span[1])
        plist: list[Posting] = []
        for i in range(offset, offset + length):
            p_count = all_pos_counts[i]
            if p_count > 0:
                positions = tuple(all_positions[pos_cursor : pos_cursor + p_count])
                pos_cursor += p_count
            else:
                positions = ()
            plist.append(Posting(doc_id=all_doc_ids[i], tf=all_tfs[i], positions=positions))
        postings_map[term] = plist

    return Index(postings=postings_map, doc_lengths=doc_lengths, doc_meta=doc_meta)


@contextmanager
def open_index(path: str | Path, fmt: FormatType = "binary") -> Generator[Index]:
    """Context manager that loads an Index and guarantees cleanup on exit."""
    index = load_index(path, fmt=fmt)
    try:
        yield index
    finally:
        index.close()


def benchmark_formats(corpus_root: str | Path, out_dir: str | Path = ".bench_tmp") -> None:
    """Compare file size, save time, and load time across pickle, json, and binary."""
    index = build_index(corpus_root)
    tmp = Path(out_dir)
    tmp.mkdir(parents=True, exist_ok=True)

    formats: list[tuple[FormatType, str]] = [
        ("pickle", "index.pkl"),
        ("json", "index.json"),
        ("binary", "index.bin"),
    ]

    header = f"{'Format':<10} | {'File size':>10} | {'Save time':>10} | {'Load time':>10}"
    print(header)
    print("-" * len(header))

    for fmt, filename in formats:
        target = tmp / filename
        t0 = time.perf_counter()
        save_index(index, target, fmt=fmt)
        save_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        _ = load_index(target, fmt=fmt)
        load_s = time.perf_counter() - t1

        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"{fmt:<10} | {size_mb:>7.2f} MB | {save_s:>8.4f} s | {load_s:>8.4f} s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark index serialization formats.")
    parser.add_argument("corpus", type=Path, help="Path to corpus directory")
    args = parser.parse_args(argv)
    benchmark_formats(args.corpus)


if __name__ == "__main__":
    main(sys.argv[1:])
