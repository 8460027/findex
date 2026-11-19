from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from findex.index import build_index
from findex.ranking import BM25, rank_documents, rank_documents_scalar


def build_corpus(path: Path, n: int = 10_000) -> None:
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
        if i % 500 == 0:
            words.append("quasar")
        lines.append(json.dumps({"id": f"doc-{i}", "title": f"Doc {i}", "text": " ".join(words)}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bench_ms(fn: object, repeats: int = 40) -> float:
    assert callable(fn)
    for _ in range(5):
        fn()
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus_10k.jsonl"
        build_corpus(corpus, n=10_000)
        idx = build_index(corpus, executor="serial")
        # РџСЂРѕРіСЂС–РІР°С”РјРѕ РјР°СЃРёРІРё С–РЅРґРµРєСЃСѓ РѕРґРёРЅ СЂР°Р· (СЏРє РїСЂРё СЃС‚Р°СЂС‚С– СЃРµСЂРІРµСЂР°)
        _ = idx.doc_lengths_array
        for term in ("python", "quasar", "bm25", "numpy"):
            idx.postings_arrays(term)

        scorer = BM25()
        cases = [
            ("common term (`python`)", ["python"]),
            ("rare term (`quasar`)", ["quasar"]),
            ("3-term query (`python`, `bm25`, `numpy`)", ["python", "bm25", "numpy"]),
        ]

        print("=== M2 Speedup Table (10,000 docs, median of 40 runs) ===")
        for label, terms in cases:
            matched = len({p.doc_id for t in terms for p in idx.postings.get(t, [])})
            t_scalar = bench_ms(lambda t=terms: rank_documents_scalar(t, idx, scorer=scorer, k=10))
            t_numpy = bench_ms(lambda t=terms: rank_documents(t, idx, scorer=scorer, k=10))
            speedup = t_scalar / max(t_numpy, 1e-9)
            print(
                f"{label} | matched={matched} | Lab3={t_scalar:.3f}ms | NumPy={t_numpy:.3f}ms | speedup={speedup:.1f}x"
            )


if __name__ == "__main__":
    main()
