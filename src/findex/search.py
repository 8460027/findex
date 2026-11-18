import argparse
import time
import tracemalloc
from pathlib import Path

from findex.index import InvertedIndex
from findex.store import load_index
from findex.tokenize import tokenize


def merge_and(a: list[int], b: list[int]) -> list[int]:
    """Two-pointer intersection of two sorted doc_id lists. O(len(a) + len(b))."""
    res: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            res.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return res


def merge_or(a: list[int], b: list[int]) -> list[int]:
    """Two-pointer union of two sorted doc_id lists. O(len(a) + len(b))."""
    res: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            res.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1
    res.extend(a[i:])
    res.extend(b[j:])
    return res


def merge_not(a: list[int], b: list[int]) -> list[int]:
    """Two-pointer difference (a \\ b) on sorted doc_id lists."""
    res: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        elif a[i] < b[j]:
            res.append(a[i])
            i += 1
        else:
            j += 1
    res.extend(a[i:])
    return res


def _get_doc_ids(index: InvertedIndex, raw_term: str) -> list[int]:
    tokens = list(tokenize(raw_term))
    if not tokens:
        return []
    plist = index.postings.get(tokens[0], [])
    return [p.doc_id for p in plist]


def evaluate_query(index: InvertedIndex, query: str, engine: str = "merge") -> list[int]:
    """Left-to-right Boolean query evaluator supporting implicit AND, OR, NOT."""
    parts = query.strip().split()
    if not parts:
        return []

    all_docs = sorted(index.doc_meta.keys())
    result: list[int] | None = None
    op = "AND"
    negate = False

    for token in parts:
        upper = token.upper()
        if upper in {"AND", "OR"}:
            op = upper
            continue
        if upper == "NOT":
            negate = not negate
            continue

        term_docs = _get_doc_ids(index, token)
        if negate:
            if engine == "merge":
                term_docs = merge_not(all_docs, term_docs)
            else:
                term_docs = sorted(set(all_docs) - set(term_docs))
            negate = False

        if result is None:
            result = term_docs
        elif engine == "merge":
            if op == "AND":
                result = merge_and(result, term_docs)
            else:
                result = merge_or(result, term_docs)
        elif engine == "set":
            if op == "AND":
                result = sorted(set(result) & set(term_docs))
            else:
                result = sorted(set(result) | set(term_docs))
        else:
            raise ValueError(f"Unknown engine: {engine}")

        op = "AND"  # reset to implicit AND

    return result or []


def benchmark_engines(index: InvertedIndex, iterations: int = 2000) -> None:
    sorted_terms = sorted(index.postings.items(), key=lambda kv: len(kv[1]), reverse=True)
    common_t1, common_t2 = sorted_terms[0][0], sorted_terms[1][0]
    rare_t1, rare_t2 = sorted_terms[-1][0], sorted_terms[-2][0]

    queries = [
        ("Most common (AND)", f"{common_t1} {common_t2}"),
        ("Most common (OR)", f"{common_t1} OR {common_t2}"),
        ("Rarest (AND)", f"{rare_t1} {rare_t2}"),
        ("Rarest (OR)", f"{rare_t1} OR {rare_t2}"),
    ]

    print(f"\nBenchmark ({iterations} runs per query):")
    print(f"Common terms: '{common_t1}' (df={len(sorted_terms[0][1])}), '{common_t2}' (df={len(sorted_terms[1][1])})")
    print(f"Rare terms:   '{rare_t1}' (df={len(sorted_terms[-1][1])}), '{rare_t2}' (df={len(sorted_terms[-2][1])})\n")
    print("| Query type | `merge` time | `set` time | Speedup |")
    print("|---|---|---|---|")

    for label, q in queries:
        t0 = time.perf_counter()
        for _ in range(iterations):
            res_m = evaluate_query(index, q, engine="merge")
        t_merge = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(iterations):
            res_s = evaluate_query(index, q, engine="set")
        t_set = time.perf_counter() - t0

        assert res_m == res_s
        ratio = t_set / t_merge if t_merge > 0 else 0
        print(f"| {label} | {t_merge * 1000:.2f} ms | {t_set * 1000:.2f} ms | {ratio:.2f}x |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search persisted inverted index.")
    parser.add_argument("index_path", type=Path, help="Path to saved index file")
    parser.add_argument("query", nargs="?", default="", help="Boolean search query")
    parser.add_argument("--format", choices=["pickle", "json", "binary"], default="pickle")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark merge vs set")
    parser.add_argument("--limit", type=int, default=10, help="Max results to display")
    args = parser.parse_args()

    tracemalloc.start()
    t0 = time.perf_counter()
    idx = load_index(args.index_path, fmt=args.format)
    load_time = time.perf_counter() - t0

    if args.benchmark:
        benchmark_engines(idx)
        return

    t1 = time.perf_counter()
    doc_ids = evaluate_query(idx, args.query, engine=args.engine)
    search_time = time.perf_counter() - t1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Found {len(doc_ids)} documents in {search_time * 1000:.3f} ms (load: {load_time:.3f} s, peak mem: {peak / 1e6:.2f} MB)")
    for did in doc_ids[: args.limit]:
        meta = idx.doc_meta[did]
        print(f"  [{did}] {meta.title} ({meta.path})")


if __name__ == "__main__":
    main()
