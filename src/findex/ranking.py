from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

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
    """Sublinear TF-IDF scorer: (1 + log(tf)) * log((N + 1) / df)."""

    def score(self, term: str, posting: Posting, index: Index) -> float:
        df = index.df(term)
        if df == 0 or posting.tf <= 0 or index.num_docs == 0:
            return 0.0
        tf_weight = 1.0 + math.log(posting.tf)
        idf = math.log((index.num_docs + 1) / df)
        return tf_weight * idf

    def score_vectorized(
        self,
        doc_ids: NDArray[np.int32],
        tfs: NDArray[np.int32],
        index: Index,
    ) -> NDArray[np.float64]:
        df = int(doc_ids.size)
        n = index.num_docs
        if df == 0 or n == 0:
            return np.empty(0, dtype=np.float64)
        idf = math.log((n + 1) / df)
        tfs_f = tfs.astype(np.float64)
        out = np.zeros(df, dtype=np.float64)
        pos = tfs_f > 0.0
        out[pos] = (1.0 + np.log(tfs_f[pos])) * idf
        return out

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

    def score_vectorized(
        self,
        doc_ids: NDArray[np.int32],
        tfs: NDArray[np.int32],
        index: Index,
    ) -> NDArray[np.float64]:
        df = int(doc_ids.size)
        n = index.num_docs
        if df == 0 or n == 0:
            return np.empty(0, dtype=np.float64)
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        avgdl = index.avg_doc_length or 1.0
        doc_lens = index.doc_lengths_array[doc_ids].astype(np.float64)
        tfs_f = tfs.astype(np.float64)
        norm = 1.0 - self.b + self.b * (doc_lens / avgdl)
        scores = idf * (tfs_f * (self.k1 + 1.0)) / (tfs_f + self.k1 * norm)
        return np.where(tfs_f > 0.0, scores, 0.0)

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)

    def __repr__(self) -> str:
        return f"BM25(k1={self.k1}, b={self.b})"


def rank_documents_scalar(
    query_terms: list[str] | tuple[str, ...],
    index: Index,
    *,
    scorer: Scorer | None = None,
    allowed_docs: set[int] | None = None,
    k: int = 10,
) -> list[tuple[int, float]]:
    """Reference Lab 03 scalar scorer loop (kept for equivalence testing and benchmarking)."""
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

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]


def rank_documents(
    query_terms: list[str] | tuple[str, ...],
    index: Index,
    *,
    scorer: Scorer | None = None,
    allowed_docs: set[int] | None = None,
    k: int = 10,
) -> list[tuple[int, float]]:
    """Vectorized NumPy scorer with O(N) np.argpartition top-k selection."""
    active_scorer: Scorer = scorer if scorer is not None else BM25()
    if not isinstance(active_scorer, (BM25, TfIdf)):
        return rank_documents_scalar(
            query_terms,
            index,
            scorer=active_scorer,
            allowed_docs=allowed_docs,
            k=k,
        )

    if k <= 0 or index.num_docs == 0:
        return []

    array_size = index.max_doc_id + 1
    if array_size <= 0:
        return []

    scores = np.zeros(array_size, dtype=np.float64)
    has_hit = np.zeros(array_size, dtype=bool)

    for term in dict.fromkeys(query_terms):
        doc_ids, tfs = index.postings_arrays(term)
        if doc_ids.size == 0:
            continue
        term_scores = active_scorer.score_vectorized(doc_ids, tfs, index)
        scores[doc_ids] += term_scores
        has_hit[doc_ids] = True

    if allowed_docs is not None:
        if not allowed_docs:
            return []
        valid_ids = [doc_id for doc_id in allowed_docs if 0 <= doc_id < array_size]
        candidate_ids = np.fromiter(valid_ids, dtype=np.int32, count=len(valid_ids))
    else:
        candidate_ids = np.flatnonzero(has_hit).astype(np.int32)

    num_candidates = int(candidate_ids.size)
    if num_candidates == 0:
        return []

    cand_scores = scores[candidate_ids]
    top_k = min(k, num_candidates)

    if top_k < num_candidates:
        # Deterministic tie-breaking by ascending doc_id inside O(N) argpartition
        tie_key = cand_scores - candidate_ids.astype(np.float64) * 1e-13
        part_idx = np.argpartition(tie_key, -top_k)[-top_k:]
        top_ids = candidate_ids[part_idx]
        top_scores = cand_scores[part_idx]
    else:
        top_ids = candidate_ids
        top_scores = cand_scores

    order = np.lexsort((top_ids, -top_scores))
    return [(int(top_ids[i]), float(top_scores[i])) for i in order]
