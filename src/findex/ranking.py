from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from operator import itemgetter
from typing import Protocol, runtime_checkable

from findex.index import Index
from findex.models import Posting


@runtime_checkable
class Scorer(Protocol):
    """Structural interface for term-document scoring functions."""

    def score(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True, order=True)
class SearchResult:
    """Ranked search result ordered naturally by score."""

    score: float
    doc_id: int
    title: str
    snippet: str = ""

    def __str__(self) -> str:
        base = f"[{self.doc_id}] {self.title} (score={self.score:.4f})"
        if self.snippet:
            return f"{base}\n      {self.snippet}"
        return base


class TfIdf:
    """Sublinear TF-IDF scorer: (1 + log(tf)) * log(N / df)."""

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or posting.tf <= 0 or index.num_docs == 0:
            return 0.0
        tf_weight = 1.0 + math.log(posting.tf)
        idf = math.log((index.num_docs + 1) / df)
        return tf_weight * idf

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)

    def __repr__(self) -> str:
        return "TfIdf()"


class BM25:
    """Okapi BM25 scorer with term-frequency saturation (k1) and length normalization (b)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or posting.tf <= 0 or index.num_docs == 0:
            return 0.0

        n = index.num_docs
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

        doc_len = index.doc_length(posting.doc_id)
        avgdl = index.avg_doc_length or 1.0
        norm = 1.0 - self.b + self.b * (doc_len / avgdl)

        tf = posting.tf
        tf_sat = (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
        return idf * tf_sat

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)

    def __repr__(self) -> str:
        return f"BM25(k1={self.k1}, b={self.b})"


def rank_documents(
    query_terms: list[str] | tuple[str, ...],
    index: Index,
    *,
    scorer: Scorer | None = None,
    allowed_docs: set[int] | None = None,
    k: int = 10,
) -> list[tuple[int, float]]:
    """Score matching documents for query terms and return top-k (doc_id, score) pairs."""
    active_scorer: Scorer = scorer if scorer is not None else BM25()
    scores: dict[int, float] = {}

    for term in dict.fromkeys(query_terms):
        if term not in index:
            continue
        for posting in index[term]:
            if allowed_docs is not None and posting.doc_id not in allowed_docs:
                continue
            scores[posting.doc_id] = scores.get(posting.doc_id, 0.0) + active_scorer.score(
                term, posting, index
            )

    if allowed_docs is not None:
        for doc_id in allowed_docs:
            scores.setdefault(doc_id, 0.0)

    return heapq.nlargest(k, scores.items(), key=itemgetter(1))
