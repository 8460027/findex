from __future__ import annotations

import argparse
import time
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import TypedDict

from findex.corpus import iter_documents
from findex.tokenize import tokenize


class CorpusStats(TypedDict):
    doc_count: int
    total_tokens: int
    vocab_size: int
    top_50: list[tuple[str, int]]
    elapsed: float
    peak_mb: float


def run_lazy_stats(root: str | Path, limit: int | None = None) -> CorpusStats:
    """Compute corpus statistics in a single lazy pass."""
    tracemalloc.start()
    t0 = time.perf_counter()

    doc_count = 0
    total_tokens = 0
    vocab: Counter[str] = Counter()

    for doc in iter_documents(root, limit=limit):
        doc_count += 1
        for tok in tokenize(doc.text):
            total_tokens += 1
            vocab[tok] += 1

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "doc_count": doc_count,
        "total_tokens": total_tokens,
        "vocab_size": len(vocab),
        "top_50": vocab.most_common(50),
        "elapsed": elapsed,
        "peak_mb": peak_bytes / (1024 * 1024),
    }


def run_eager_stats(root: str | Path, limit: int | None = None) -> CorpusStats:
    """Deliberately materialize all documents and token lists in memory."""
    tracemalloc.start()
    t0 = time.perf_counter()

    docs_list = list(iter_documents(root, limit=limit))
    all_tokens = [list(tokenize(doc.text)) for doc in docs_list]

    vocab: Counter[str] = Counter()
    total_tokens = 0
    for doc_tokens in all_tokens:
        total_tokens += len(doc_tokens)
        vocab.update(doc_tokens)

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "doc_count": len(docs_list),
        "total_tokens": total_tokens,
        "vocab_size": len(vocab),
        "top_50": vocab.most_common(50),
        "elapsed": elapsed,
        "peak_mb": peak_bytes / (1024 * 1024),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream corpus statistics.")
    parser.add_argument("corpus", type=Path, help="Path to corpus directory or .jsonl file")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N documents")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run both eager and lazy pipelines and print Markdown comparison table",
    )
    args = parser.parse_args()

    if args.benchmark:
        eager = run_eager_stats(args.corpus, args.limit)
        lazy = run_lazy_stats(args.corpus, args.limit)

        print("| Version | Documents | Peak memory | Elapsed |")
        print("|---|---|---|---|")
        print(
            f"| eager (lists) | {eager['doc_count']} | "
            f"{eager['peak_mb']:.2f} MB | {eager['elapsed']:.3f} s |"
        )
        print(
            f"| lazy (generators) | {lazy['doc_count']} | "
            f"{lazy['peak_mb']:.2f} MB | {lazy['elapsed']:.3f} s |"
        )
        return

    res = run_lazy_stats(args.corpus, args.limit)
    print(f"Documents:       {res['doc_count']}")
    print(f"Total tokens:    {res['total_tokens']}")
    print(f"Vocabulary size: {res['vocab_size']}")
    print(f"Elapsed time:    {res['elapsed']:.3f} s")
    print(f"Peak memory:     {res['peak_mb']:.2f} MB\n")
    print("Top 50 terms:")
    for rank, (term, count) in enumerate(res["top_50"], start=1):
        print(f"{rank:>2}. {term:<20} {count}")


if __name__ == "__main__":
    main()
