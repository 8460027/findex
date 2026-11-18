from __future__ import annotations

import argparse
import functools
import pickle
import sys
import tempfile
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents
from findex.decorators import timed
from findex.models import DocMeta, Posting
from findex.tokenize import tokenize


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


@timed
def build_index(
    root: str | Path,
    *,
    with_positions: bool = False,
    limit: int | None = None,
) -> Index:
    """Build an inverted index in a single streaming pass over the corpus."""
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, (doc_name, doc_path, text) in enumerate(iter_documents(root)):
        tokens = list(tokenize(text))
        doc_lengths[doc_id] = len(tokens)
        doc_meta[doc_id] = DocMeta(doc_id=doc_id, title=doc_name, path=str(doc_path))

        if with_positions:
            pos_map: defaultdict[str, list[int]] = defaultdict(list)
            for pos, tok in enumerate(tokens):
                pos_map[tok].append(pos)
            for term, positions in pos_map.items():
                postings[term].append(
                    Posting(doc_id=doc_id, tf=len(positions), positions=tuple(positions))
                )
        else:
            counts = Counter(tokens)
            for term, tf in counts.items():
                postings[term].append(Posting(doc_id=doc_id, tf=tf))

    return Index(
        postings=dict(postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
    )


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
    index = build_index(args.corpus, with_positions=args.positions)
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
