from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from findex.index import Index, build_index
from findex.ranking import BM25, TfIdf, rank_documents, rank_documents_scalar
from findex.search import SearchEngine


@pytest.fixture(scope="module")
def bench_corpus_and_index(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Index]:
    tmp = tmp_path_factory.mktemp("bench_lab08")
    corpus = tmp / "corpus.jsonl"
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
    for i in range(600):
        words = [vocab[(i + j * 5) % len(vocab)] for j in range(35 + (i % 19))]
        if i % 40 == 0:
            words.append("rare_quantum_token")
        lines.append(json.dumps({"id": f"doc-{i}", "title": f"Doc {i}", "text": " ".join(words)}))
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    idx = build_index(corpus, executor="serial")
    return corpus, idx


@pytest.mark.parametrize("scorer_cls", [BM25, TfIdf])
@pytest.mark.parametrize(
    "terms",
    [
        ["python"],
        ["rare_quantum_token"],
        ["python", "bm25", "numpy"],
        ["asyncio", "fastapi", "nonexistent_term"],
    ],
)
def test_numpy_scorer_matches_lab3_scalar_ranking(
    bench_corpus_and_index: tuple[Path, Index],
    scorer_cls: type[BM25 | TfIdf],
    terms: list[str],
) -> None:
    _, idx = bench_corpus_and_index
    scorer = scorer_cls()

    scalar_res = rank_documents_scalar(terms, idx, scorer=scorer, k=15)
    numpy_res = rank_documents(terms, idx, scorer=scorer, k=15)

    assert len(numpy_res) == len(scalar_res)
    for (doc_np, score_np), (doc_sc, score_sc) in zip(numpy_res, scalar_res, strict=True):
        assert np.isclose(score_np, score_sc, rtol=1e-12, atol=1e-12)
        if abs(score_np - score_sc) <= 1e-12:
            assert doc_np == doc_sc


def test_numpy_scorer_with_allowed_docs_matches_scalar(
    bench_corpus_and_index: tuple[Path, Index],
) -> None:
    _, idx = bench_corpus_and_index
    allowed = {0, 3, 7, 15, 40, 120, 599}
    scorer = BM25()

    scalar_res = rank_documents_scalar(
        ["python", "bm25"], idx, scorer=scorer, allowed_docs=allowed, k=5
    )
    numpy_res = rank_documents(["python", "bm25"], idx, scorer=scorer, allowed_docs=allowed, k=5)

    assert len(numpy_res) == len(scalar_res)
    for (doc_np, score_np), (doc_sc, score_sc) in zip(numpy_res, scalar_res, strict=True):
        assert doc_np == doc_sc
        assert np.isclose(score_np, score_sc, rtol=1e-12, atol=1e-12)


def test_benchmark_search(benchmark: Any, bench_corpus_and_index: tuple[Path, Index]) -> None:
    _, idx = bench_corpus_and_index
    engine = SearchEngine(idx)
    hits = benchmark(lambda: engine._compute_ranked("python OR bm25 OR numpy", "bm25", 10))
    assert len(hits) == 10


def test_benchmark_build_index(benchmark: Any, bench_corpus_and_index: tuple[Path, Index]) -> None:
    corpus, _ = bench_corpus_and_index
    idx = benchmark(lambda: build_index(corpus, executor="serial"))
    assert idx.num_docs == 600
