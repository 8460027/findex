from __future__ import annotations

import cProfile
import io
import json
import pstats
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from findex.index import build_index
from findex.search import SearchEngine


def build_corpus(path: Path, n: int = 5000) -> None:
    vocab = [
        "python",
        "asyncio",
        "event",
        "loop",
        "coroutine",
        "fastapi",
        "pydantic",
        "bm25",
        "tfidf",
        "inverted",
        "index",
        "postings",
        "ranking",
        "query",
        "multiprocessing",
        "gil",
        "threadpool",
        "numpy",
        "vector",
        "embedding",
    ]
    lines: list[str] = []
    for i in range(n):
        words = [vocab[(i + j * 7) % len(vocab)] for j in range(60)]
        lines.append(json.dumps({"id": f"doc-{i}", "title": f"Doc {i}", "text": " ".join(words)}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus_5k.jsonl"
        build_corpus(corpus, n=5000)

        pr_idx = cProfile.Profile()
        pr_idx.enable()
        idx = build_index(corpus, executor="serial")
        pr_idx.disable()

        s_idx = io.StringIO()
        pstats.Stats(pr_idx, stream=s_idx).sort_stats("tottime").print_stats(10)
        print("=== cProfile TOP (build_index, sorted by tottime) ===")
        print(s_idx.getvalue())

        engine = SearchEngine(idx)
        pr_search = cProfile.Profile()
        pr_search.enable()
        for i in range(250):
            q = f"python OR bm25 OR numpy OR doc_{i}"
            engine._compute_ranked(q, "bm25", 10)
        pr_search.disable()

        s_search = io.StringIO()
        pstats.Stats(pr_search, stream=s_search).sort_stats("tottime").print_stats(10)
        print("=== cProfile TOP (250x 3+ term BM25 search, sorted by tottime) ===")
        print(s_search.getvalue())


if __name__ == "__main__":
    main()
