import argparse
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents
from findex.models import DocMeta, InvertedIndex, Posting
from findex.tokenize import tokenize


def build_index(root: Path, with_positions: bool = False) -> InvertedIndex:
    postings: dict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc_id, doc in enumerate(iter_documents(root)):
        doc_meta[doc_id] = DocMeta(doc_id=doc_id, path=str(doc.path), title=getattr(doc, "title", Path(doc.path).stem))
        tokens = list(tokenize(doc.text))
        doc_lengths[doc_id] = len(tokens)

        if with_positions:
            pos_map: dict[str, list[int]] = defaultdict(list)
            for idx, tok in enumerate(tokens):
                pos_map[tok].append(idx)
            for term, pos_list in pos_map.items():
                postings[term].append(
                    Posting(doc_id=doc_id, tf=len(pos_list), positions=tuple(pos_list))
                )
        else:
            counts = Counter(tokens)
            for term, tf in counts.items():
                postings[term].append(Posting(doc_id=doc_id, tf=tf))

    return InvertedIndex(
        postings=dict(postings),
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        with_positions=with_positions,
    )


# --- M4: Варіанти для дослідження пам'яті ---

@dataclass
class PlainPosting:
    doc_id: int
    tf: int


def build_plain_postings(root: Path) -> dict[str, list[PlainPosting]]:
    postings: dict[str, list[PlainPosting]] = defaultdict(list)
    for doc_id, doc in enumerate(iter_documents(root)):
        counts = Counter(tokenize(doc.text))
        for term, tf in counts.items():
            postings[term].append(PlainPosting(doc_id=doc_id, tf=tf))
    return dict(postings)


def build_array_postings(root: Path) -> dict[str, tuple[array, array]]:
    postings: dict[str, tuple[array, array]] = defaultdict(
        lambda: (array("I"), array("I"))
    )
    for doc_id, doc in enumerate(iter_documents(root)):
        counts = Counter(tokenize(doc.text))
        for term, tf in counts.items():
            doc_ids, tfs = postings[term]
            doc_ids.append(doc_id)
            tfs.append(tf)
    return dict(postings)


def run_memory_study(root: Path) -> None:
    from findex.store import load_pickle, save_pickle

    tmp_dir = Path(".bench_tmp")
    tmp_dir.mkdir(exist_ok=True)

    results = []

    # 1. Plain @dataclass
    tracemalloc.start()
    plain = build_plain_postings(root)
    _, peak_plain = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    p1 = tmp_dir / "plain.pkl"
    save_pickle(plain, p1)
    t0 = time.perf_counter()
    _ = load_pickle(p1)
    t_plain = time.perf_counter() - t0
    results.append(("`list[Posting]` with plain `@dataclass`", peak_plain, p1.stat().st_size, t_plain))
    del plain

    # 2. @dataclass(slots=True)
    tracemalloc.start()
    slotted = build_index(root, with_positions=False)
    _, peak_slotted = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    p2 = tmp_dir / "slotted.pkl"
    save_pickle(slotted.postings, p2)
    t0 = time.perf_counter()
    _ = load_pickle(p2)
    t_slotted = time.perf_counter() - t0
    results.append(("`list[Posting]` with `slots=True`", peak_slotted, p2.stat().st_size, t_slotted))
    del slotted

    # 3. array('I') pairs
    tracemalloc.start()
    arr = build_array_postings(root)
    _, peak_arr = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    p3 = tmp_dir / "array.pkl"
    save_pickle(arr, p3)
    t0 = time.perf_counter()
    _ = load_pickle(p3)
    t_arr = time.perf_counter() - t0
    results.append(("`array('I')` pairs (doc_ids, tfs)", peak_arr, p3.stat().st_size, t_arr))
    del arr

    print("\n| Postings representation | Peak memory (build) | Index file size | Load time |")
    print("|---|---|---|---|")
    for name, mem, size, lt in results:
        print(f"| {name} | {mem / 1e6:.2f} MB | {size / 1e6:.2f} MB | {lt:.4f} s |")

    # Вимірюємо вартість positions
    tracemalloc.start()
    idx_pos = build_index(root, with_positions=True)
    _, peak_pos = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    p_pos = tmp_dir / "pos.pkl"
    save_pickle(idx_pos, p_pos)
    print(f"\nWith positions=True: Peak memory = {peak_pos / 1e6:.2f} MB, File size = {p_pos.stat().st_size / 1e6:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist inverted index.")
    parser.add_argument("corpus", type=Path, help="Path to corpus directory")
    parser.add_argument("--out", type=Path, default=Path("index.bin"), help="Output index file")
    parser.add_argument("--format", choices=["pickle", "json", "binary"], default="pickle")
    parser.add_argument("--positions", action="store_true", help="Store token positions")
    parser.add_argument("--memory-study", action="store_true", help="Run M4 memory comparison")
    args = parser.parse_args()

    if args.memory_study:
        run_memory_study(args.corpus)
        return

    from findex.store import save_index

    tracemalloc.start()
    t0 = time.perf_counter()
    idx = build_index(args.corpus, with_positions=args.positions)
    build_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    save_time = save_index(idx, args.out, fmt=args.format)
    size_mb = args.out.stat().st_size / 1e6

    print(f"Indexed {len(idx.doc_meta)} docs, {len(idx.postings)} unique terms.")
    print(f"Build time: {build_time:.3f} s | Peak memory: {peak / 1e6:.2f} MB")
    print(f"Saved to {args.out} ({args.format}): {size_mb:.2f} MB in {save_time:.3f} s")


if __name__ == "__main__":
    main()


