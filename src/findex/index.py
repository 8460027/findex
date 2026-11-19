from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import pickle
import sys
import tempfile
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypedDict, cast

from findex.corpus import iter_documents
from findex.decorators import timed
from findex.models import DocMeta, PartialIndex, Posting
from findex.tokenize import tokenize

log = logging.getLogger(__name__)

ExecutorType = Literal["serial", "threads", "processes"]


class DocRef(NamedTuple):
    """Lightweight, picklable reference to a document on disk."""

    doc_id: int
    root_str: str
    path_str: str
    line_no: int | None = None


class BuildTiming(TypedDict):
    map_seconds: float
    merge_seconds: float
    total_seconds: float


class Index(Mapping[str, list[Posting]]):
    """Inverted index behaving as a Python Mapping[str, list[Posting]]."""

    def __init__(
        self,
        postings: dict[str, list[Posting]] | None = None,
        doc_lengths: dict[int, int] | None = None,
        doc_meta: dict[int, DocMeta] | None = None,
    ) -> None:
        self.postings: dict[str, list[Posting]] = postings if postings is not None else {}
        self.doc_lengths: dict[int, int] = doc_lengths if doc_lengths is not None else {}
        self.doc_meta: dict[int, DocMeta] = doc_meta if doc_meta is not None else {}
        self._closed: bool = False

    def __len__(self) -> int:
        return len(self.postings)

    def __contains__(self, term: object) -> bool:
        return term in self.postings

    def __getitem__(self, term: str) -> list[Posting]:
        return self.postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @functools.cached_property
    def avg_doc_length(self) -> float:
        if self.num_docs == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / self.num_docs

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        postings = self.postings.get(term)
        return len(postings) if postings is not None else 0

    def close(self) -> None:
        """Release index resources."""
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed


InvertedIndex = Index


@dataclass
class PlainPosting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


def collect_doc_refs(root: str | Path, *, limit: int | None = None) -> list[DocRef]:
    """Scan corpus paths up front and assign globally unique doc_ids without loading full texts."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root_path}")

    def _iter_refs() -> Iterator[DocRef]:
        next_id = 0
        paths = [root_path] if root_path.is_file() else sorted(root_path.rglob("*"))
        root_str = str(root_path)
        for path in paths:
            if not path.is_file():
                continue
            if path.suffix == ".jsonl":
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    for line_no, raw_line in enumerate(f, start=1):
                        if raw_line.strip():
                            yield DocRef(
                                doc_id=next_id,
                                root_str=root_str,
                                path_str=str(path),
                                line_no=line_no,
                            )
                            next_id += 1
            elif path.suffix in {".txt", ".md"}:
                yield DocRef(
                    doc_id=next_id,
                    root_str=root_str,
                    path_str=str(path),
                    line_no=None,
                )
                next_id += 1

    stream = _iter_refs()
    if limit is not None:
        return list(islice(stream, limit))
    return list(stream)


def _load_doc_ref(ref: DocRef) -> tuple[str, str] | None:
    """Read a single document from disk given its lightweight DocRef. Returns (title, text)."""
    path = Path(ref.path_str)
    root = Path(ref.root_str)
    if ref.line_no is None:
        try:
            raw_bytes = path.read_bytes()
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                log.warning("Invalid UTF-8 in %s; decoding with replacement.", path)
                text = raw_bytes.decode("utf-8", errors="replace")
            title = str(path.relative_to(root)) if root.is_dir() else path.name
            return title, text
        except OSError as exc:
            log.error("Failed to read %s: %s", path, exc)
            return None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for current_line_no, raw_line in enumerate(f, start=1):
            if current_line_no != ref.line_no:
                continue
            line = raw_line.strip()
            if not line:
                return None
            try:
                loaded: object = json.loads(line)
                if not isinstance(loaded, dict):
                    log.warning("Expected JSON object at %s:%d; skipping.", path, ref.line_no)
                    return None
                record = cast(dict[str, Any], loaded)
                text = str(record.get("text") or record.get("abstract") or "")
                title = str(record.get("id", f"{path.name}:{ref.line_no}"))
                return title, text
            except json.JSONDecodeError:
                log.warning("Malformed JSON at %s:%d; skipping.", path, ref.line_no)
                return None
    return None


def build_partial(
    doc_refs: list[DocRef],
    *,
    with_positions: bool = False,
) -> PartialIndex:
    """Map step: build a compact PartialIndex for a chunk of document paths (picklable)."""
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    # Group JSONL references by file path so multi-line JSONL files are read in one linear scan
    jsonl_targets: dict[str, dict[int, int]] = defaultdict(dict)
    ordered_refs: list[DocRef] = []
    for ref in doc_refs:
        ordered_refs.append(ref)
        if ref.line_no is not None:
            jsonl_targets[ref.path_str][ref.line_no] = ref.doc_id

    jsonl_cache: dict[int, tuple[str, str] | None] = {}
    for path_str, line_to_doc_id in jsonl_targets.items():
        path = Path(path_str)
        max_line = max(line_to_doc_id)
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, raw_line in enumerate(f, start=1):
                if line_no > max_line:
                    break
                doc_id = line_to_doc_id.get(line_no)
                if doc_id is None:
                    continue
                line = raw_line.strip()
                if not line:
                    jsonl_cache[doc_id] = None
                    continue
                try:
                    loaded: object = json.loads(line)
                    if not isinstance(loaded, dict):
                        jsonl_cache[doc_id] = None
                        continue
                    record = cast(dict[str, Any], loaded)
                    text = str(record.get("text") or record.get("abstract") or "")
                    title = str(record.get("id", f"{path.name}:{line_no}"))
                    jsonl_cache[doc_id] = (title, text)
                except json.JSONDecodeError:
                    jsonl_cache[doc_id] = None

    for ref in ordered_refs:
        loaded_doc = jsonl_cache.get(ref.doc_id) if ref.line_no is not None else _load_doc_ref(ref)
        if loaded_doc is None:
            continue
        title, text = loaded_doc
        tokens = list(tokenize(text))
        doc_lengths[ref.doc_id] = len(tokens)
        doc_meta[ref.doc_id] = DocMeta(doc_id=ref.doc_id, title=title, path=ref.path_str)

        if with_positions:
            pos_map: defaultdict[str, list[int]] = defaultdict(list)
            for pos, tok in enumerate(tokens):
                pos_map[tok].append(pos)
            for term, positions in pos_map.items():
                postings[term].append(
                    Posting(doc_id=ref.doc_id, tf=len(positions), positions=tuple(positions))
                )
        else:
            counts = Counter(tokens)
            for term, tf in counts.items():
                postings[term].append(Posting(doc_id=ref.doc_id, tf=tf))

    return PartialIndex(
        postings=dict(postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
    )


def merge(partials: Iterable[PartialIndex]) -> Index:
    """Reduce step: sequentially merge partial indexes into a single sorted Index."""
    merged_postings: defaultdict[str, list[Posting]] = defaultdict(list)
    merged_lengths: dict[int, int] = {}
    merged_meta: dict[int, DocMeta] = {}

    for partial in partials:
        merged_lengths.update(partial.doc_lengths)
        merged_meta.update(partial.doc_meta)
        for term, plist in partial.postings.items():
            merged_postings[term].extend(plist)

    for plist in merged_postings.values():
        if len(plist) > 1:
            plist.sort(key=lambda p: p.doc_id)

    return Index(
        postings={term: merged_postings[term] for term in sorted(merged_postings)},
        doc_lengths=dict(sorted(merged_lengths.items())),
        doc_meta=dict(sorted(merged_meta.items())),
    )


def _chunk_refs(refs: list[DocRef], num_chunks: int) -> list[list[DocRef]]:
    if not refs:
        return []
    n = max(1, min(num_chunks, len(refs)))
    chunk_size = (len(refs) + n - 1) // n
    return [refs[i : i + chunk_size] for i in range(0, len(refs), chunk_size)]


@timed
def build_index(
    root: str | Path,
    *,
    with_positions: bool = False,
    limit: int | None = None,
    executor: ExecutorType = "serial",
    workers: int | None = None,
    timings_out: BuildTiming | None = None,
) -> Index:
    """Build an inverted index using serial, threaded, or multi-process map-reduce."""
    t_start = time.perf_counter()
    refs = collect_doc_refs(root, limit=limit)

    if not refs:
        if timings_out is not None:
            timings_out["map_seconds"] = 0.0
            timings_out["merge_seconds"] = 0.0
            timings_out["total_seconds"] = time.perf_counter() - t_start
        return Index()

    worker_count = workers if workers is not None and workers > 0 else (os.cpu_count() or 1)

    t_map_start = time.perf_counter()
    if executor == "serial" or worker_count == 1 or len(refs) == 1:
        partials: list[PartialIndex] = [build_partial(refs, with_positions=with_positions)]
    elif executor == "threads":
        chunks = _chunk_refs(refs, worker_count)
        partials = []
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [
                pool.submit(build_partial, chunk, with_positions=with_positions) for chunk in chunks
            ]
            for fut in as_completed(futures):
                partials.append(fut.result())
    elif executor == "processes":
        chunks = _chunk_refs(refs, worker_count)
        partials = []
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [
                pool.submit(build_partial, chunk, with_positions=with_positions) for chunk in chunks
            ]
            for fut in as_completed(futures):
                partials.append(fut.result())
    else:
        raise ValueError(f"Unsupported executor: {executor}")

    map_seconds = time.perf_counter() - t_map_start

    t_merge_start = time.perf_counter()
    result = merge(partials)
    merge_seconds = time.perf_counter() - t_merge_start
    total_seconds = time.perf_counter() - t_start

    if timings_out is not None:
        timings_out["map_seconds"] = map_seconds
        timings_out["merge_seconds"] = merge_seconds
        timings_out["total_seconds"] = total_seconds

    log.info(
        "Built index (%s, workers=%d): map=%.3fs, merge=%.3fs, total=%.3fs",
        executor,
        worker_count,
        map_seconds,
        merge_seconds,
        total_seconds,
    )
    return result


def _build_plain_dataclass(root: str | Path) -> dict[str, list[PlainPosting]]:
    postings: defaultdict[str, list[PlainPosting]] = defaultdict(list)
    for doc_id, (_, _, text) in enumerate(iter_documents(root)):
        for term, tf in Counter(tokenize(text)).items():
            postings[term].append(PlainPosting(doc_id=doc_id, tf=tf))
    return dict(postings)


def _build_array_pairs(root: str | Path) -> dict[str, tuple[array[int], array[int]]]:
    postings: dict[str, tuple[array[int], array[int]]] = {}
    for doc_id, (_, _, text) in enumerate(iter_documents(root)):
        for term, tf in Counter(tokenize(text)).items():
            pair = postings.get(term)
            if pair is None:
                pair = (array("I"), array("I"))
                postings[term] = pair
            pair[0].append(doc_id)
            pair[1].append(tf)
    return postings


def run_memory_study(root: str | Path) -> None:
    """Measure peak memory, disk footprint, and load speed across 3 representations."""
    print(f"Running memory study on corpus: {root}\n")
    results: list[tuple[str, float, float, float]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        tracemalloc.start()
        plain_idx = _build_plain_dataclass(root)
        _, peak_plain = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        p_plain = tmp / "plain.pkl"
        with p_plain.open("wb") as f:
            pickle.dump(plain_idx, f, protocol=pickle.HIGHEST_PROTOCOL)
        t0 = time.perf_counter()
        with p_plain.open("rb") as f:
            _ = pickle.load(f)
        load_plain = time.perf_counter() - t0
        results.append(
            (
                "list[Posting] (plain @dataclass)",
                peak_plain / (1024 * 1024),
                p_plain.stat().st_size / (1024 * 1024),
                load_plain,
            )
        )
        del plain_idx

        tracemalloc.start()
        slots_idx = build_index(root, with_positions=False)
        _, peak_slots = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        p_slots = tmp / "slots.pkl"
        with p_slots.open("wb") as f:
            pickle.dump(slots_idx.postings, f, protocol=pickle.HIGHEST_PROTOCOL)
        t0 = time.perf_counter()
        with p_slots.open("rb") as f:
            _ = pickle.load(f)
        load_slots = time.perf_counter() - t0
        results.append(
            (
                "list[Posting] (slots=True)",
                peak_slots / (1024 * 1024),
                p_slots.stat().st_size / (1024 * 1024),
                load_slots,
            )
        )
        del slots_idx

        tracemalloc.start()
        arr_idx = _build_array_pairs(root)
        _, peak_arr = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        p_arr = tmp / "array.pkl"
        with p_arr.open("wb") as f:
            pickle.dump(arr_idx, f, protocol=pickle.HIGHEST_PROTOCOL)
        t0 = time.perf_counter()
        with p_arr.open("rb") as f:
            _ = pickle.load(f)
        load_arr = time.perf_counter() - t0
        results.append(
            (
                "array('I') pairs (doc_ids, tfs)",
                peak_arr / (1024 * 1024),
                p_arr.stat().st_size / (1024 * 1024),
                load_arr,
            )
        )
        del arr_idx

        tracemalloc.start()
        pos_idx = build_index(root, with_positions=True)
        _, peak_pos = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        p_pos = tmp / "positions.pkl"
        with p_pos.open("wb") as f:
            pickle.dump(pos_idx.postings, f, protocol=pickle.HIGHEST_PROTOCOL)
        pos_size_mb = p_pos.stat().st_size / (1024 * 1024)

    header = f"{'Postings representation':<35} | {'Peak mem':>10} | {'File size':>10} | {'Load time':>10}"
    print(header)
    print("-" * len(header))
    for name, mem_mb, size_mb, load_s in results:
        print(f"{name:<35} | {mem_mb:>7.2f} MB | {size_mb:>7.2f} MB | {load_s:>8.4f} s")

    print(
        f"\nOptional positions cost (--positions): "
        f"Peak mem = {peak_pos / (1024 * 1024):.2f} MB, File size = {pos_size_mb:.2f} MB"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build an inverted index from a corpus.")
    parser.add_argument("corpus", type=Path, help="Path to corpus directory")
    parser.add_argument("--out", type=Path, default=Path("index.bin"), help="Output index path")
    parser.add_argument(
        "--format",
        choices=["pickle", "json", "binary"],
        default="binary",
        help="Serialization format",
    )
    parser.add_argument("--positions", action="store_true", help="Store token positions")
    parser.add_argument(
        "--executor",
        choices=["serial", "threads", "processes"],
        default="processes",
        help="Execution backend",
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of worker threads/processes"
    )
    parser.add_argument(
        "--memory-study",
        action="store_true",
        help="Run memory comparison across postings representations",
    )
    args = parser.parse_args(argv)

    if args.memory_study:
        run_memory_study(args.corpus)
        return

    from findex.store import save_index

    tracemalloc.start()
    t0 = time.perf_counter()
    index = build_index(
        args.corpus,
        with_positions=args.positions,
        executor=args.executor,
        workers=args.workers,
    )
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Built {index!r} in {elapsed:.3f} s | Peak memory: {peak / (1024 * 1024):.2f} MB")

    t_save = time.perf_counter()
    save_index(index, args.out, fmt=args.format)
    save_elapsed = time.perf_counter() - t_save
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"Saved to {args.out} ({args.format}): {size_mb:.2f} MB in {save_elapsed:.3f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
