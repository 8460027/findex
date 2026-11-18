from __future__ import annotations

import argparse
import functools
import logging
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Literal, NamedTuple


class CacheStats(NamedTuple):
    hits: int
    misses: int
    maxsize: int
    currsize: int


from findex.decorators import timed
from findex.index import Index
from findex.models import Posting
from findex.query import parse_query
from findex.ranking import BM25, Scorer, SearchResult, TfIdf, rank_documents
from findex.snippets import build_snippet
from findex.store import open_index
from findex.tokenize import tokenize

log = logging.getLogger("findex")


def intersect_postings(a: list[Posting], b: list[Posting]) -> list[int]:
    """Two-pointer sorted postings intersection (AND) in O(|a| + |b|)."""
    i = 0
    j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        da = a[i].doc_id
        db = b[j].doc_id
        if da == db:
            out.append(da)
            i += 1
            j += 1
        elif da < db:
            i += 1
        else:
            j += 1
    return out


def union_postings(a: list[Posting], b: list[Posting]) -> list[int]:
    """Two-pointer sorted postings union (OR) in O(|a| + |b|)."""
    i = 0
    j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        da = a[i].doc_id
        db = b[j].doc_id
        if da == db:
            out.append(da)
            i += 1
            j += 1
        elif da < db:
            out.append(da)
            i += 1
        else:
            out.append(db)
            j += 1
    while i < len(a):
        out.append(a[i].doc_id)
        i += 1
    while j < len(b):
        out.append(b[j].doc_id)
        j += 1
    return out


def boolean_search_merge(query: str, index: Index) -> list[int]:
    """Evaluate simple Boolean query using two-pointer merge."""
    tokens = query.strip().split()
    if not tokens:
        return []

    op: Literal["AND", "OR"] = "AND"
    terms: list[str] = []
    for tok in tokens:
        if tok in ("AND", "OR"):
            op = tok
        else:
            terms.extend(tokenize(tok))

    if not terms:
        return []

    first_postings = index.get(terms[0], [])
    if len(terms) == 1:
        return [p.doc_id for p in first_postings]

    result_ids = [p.doc_id for p in first_postings]
    for term in terms[1:]:
        next_postings = index.get(term, [])
        if op == "AND":
            cur_postings = [Posting(doc_id=d, tf=1) for d in result_ids]
            result_ids = intersect_postings(cur_postings, next_postings)
        else:
            cur_postings = [Posting(doc_id=d, tf=1) for d in result_ids]
            result_ids = union_postings(cur_postings, next_postings)
    return result_ids


def boolean_search_set(query: str, index: Index) -> list[int]:
    """Evaluate simple Boolean query using Python built-in set operations."""
    tokens = query.strip().split()
    if not tokens:
        return []

    op: Literal["AND", "OR"] = "AND"
    terms: list[str] = []
    for tok in tokens:
        if tok in ("AND", "OR"):
            op = tok
        else:
            terms.extend(tokenize(tok))

    if not terms:
        return []

    result_set = {p.doc_id for p in index.get(terms[0], [])}
    for term in terms[1:]:
        term_set = {p.doc_id for p in index.get(term, [])}
        if op == "AND":
            result_set &= term_set
        else:
            result_set |= term_set
    return sorted(result_set)


def evaluate_query(
    index: Index, query: str, engine: Literal["merge", "set"] = "merge"
) -> list[int]:
    """Backward-compatible Boolean query evaluator for Lab 02."""
    if "NOT" in query or "(" in query or '"' in query:
        return sorted(parse_query(query).evaluate(index))
    if engine == "merge":
        return boolean_search_merge(query, index)
    return boolean_search_set(query, index)


class SearchEngine:
    """Ranked search engine with LRU query caching and snippet generation."""

    def __init__(self, index: Index) -> None:
        self.index = index
        self._scorers: dict[str, Scorer] = {
            "bm25": BM25(),
            "tfidf": TfIdf(),
        }
        self._cached_rank = functools.lru_cache(maxsize=256)(self._compute_ranked)

    def _register_scorer(self, scorer: Scorer | str | None) -> str:
        if scorer is None or scorer == "bm25":
            return "bm25"
        if scorer == "tfidf":
            return "tfidf"
        if not isinstance(scorer, Scorer):
            msg = f"Unknown scorer: {scorer}"
            raise TypeError(msg)
        key = repr(scorer)
        self._scorers[key] = scorer
        return key

    def _compute_ranked(
        self, query_str: str, scorer_key: str, k: int
    ) -> tuple[tuple[int, float], ...]:
        ast = parse_query(query_str)
        matched_docs = ast.evaluate(self.index)
        if not matched_docs:
            return ()
        scoring_terms = ast.positive_terms()
        ranked = rank_documents(
            scoring_terms,
            self.index,
            scorer=self._scorers[scorer_key],
            allowed_docs=matched_docs,
            k=k,
        )
        return tuple(ranked)

    @timed
    def search(
        self,
        query: str,
        *,
        scorer: Scorer | str | None = None,
        k: int = 10,
        with_snippets: bool = False,
    ) -> list[SearchResult]:
        """Parse query, evaluate AST, rank matched documents, and optionally attach snippets."""
        scorer_key = self._register_scorer(scorer)
        ranked_pairs = self._cached_rank(query, scorer_key, k)

        terms_for_snippet = parse_query(query).positive_terms() if with_snippets else []
        results: list[SearchResult] = []
        for doc_id, score in ranked_pairs:
            meta = self.index.doc_meta.get(doc_id)
            title = meta.title if meta is not None else f"doc_{doc_id}"
            snippet = (
                build_snippet(meta.path, terms_for_snippet)
                if (with_snippets and meta is not None)
                else ""
            )
            results.append(SearchResult(score=score, doc_id=doc_id, title=title, snippet=snippet))
        return results

    def cache_info(self) -> CacheStats:
        info = self._cached_rank.cache_info()
        return CacheStats(info.hits, info.misses, info.maxsize or 0, info.currsize)


def run_benchmark(index: Index, iterations: int = 2000) -> None:
    """Compare two-pointer merge vs set intersection/union on frequent and rare terms."""
    sorted_terms = sorted(index.postings.items(), key=lambda kv: len(kv[1]), reverse=True)
    common_t1, common_t2 = sorted_terms[0][0], sorted_terms[1][0]
    rare_t1, rare_t2 = sorted_terms[-1][0], sorted_terms[-2][0]

    queries = [
        ("Most common (AND)", f"{common_t1} AND {common_t2}"),
        ("Most common (OR)", f"{common_t1} OR {common_t2}"),
        ("Rarest (AND)", f"{rare_t1} AND {rare_t2}"),
        ("Rarest (OR)", f"{rare_t1} OR {rare_t2}"),
    ]

    print(
        f"Benchmarking {iterations} iterations per query:\n"
        f"  Common terms: {common_t1!r} (df={index.df(common_t1)}), "
        f"{common_t2!r} (df={index.df(common_t2)})\n"
        f"  Rare terms:   {rare_t1!r} (df={index.df(rare_t1)}), "
        f"{rare_t2!r} (df={index.df(rare_t2)})\n"
    )

    header = f"{'Query type':<22} | {'merge (ms)':>12} | {'set (ms)':>12} | {'Speedup':>10}"
    print(header)
    print("-" * len(header))

    for label, q in queries:
        t0 = time.perf_counter()
        for _ in range(iterations):
            boolean_search_merge(q, index)
        merge_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        for _ in range(iterations):
            boolean_search_set(q, index)
        set_ms = (time.perf_counter() - t1) * 1000

        speedup = set_ms / merge_ms if merge_ms > 0 else 0.0
        print(f"{label:<22} | {merge_ms:>12.2f} | {set_ms:>12.2f} | {speedup:>9.2f}x")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search an inverted index.")
    parser.add_argument("index_path", type=Path, help="Path to saved index file")
    parser.add_argument("query", nargs="?", default="", help="Search query")
    parser.add_argument(
        "--format",
        choices=["pickle", "json", "binary"],
        default="binary",
        help="Index file format",
    )
    parser.add_argument(
        "--mode",
        choices=["ranked", "boolean"],
        default="ranked",
        help="Search mode (ranked TF-IDF/BM25 or legacy boolean)",
    )
    parser.add_argument(
        "--scorer",
        choices=["bm25", "tfidf"],
        default="bm25",
        help="Ranking scorer (for ranked mode)",
    )
    parser.add_argument(
        "--engine",
        choices=["merge", "set"],
        default="merge",
        help="Boolean engine (for legacy boolean mode)",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k results to display")
    parser.add_argument("--snippets", action="store_true", help="Show highlighted snippets")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run merge vs set benchmark",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable timing logs")
    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tracemalloc.start()
    t_load = time.perf_counter()
    with open_index(args.index_path, fmt=args.format) as index:
        load_s = time.perf_counter() - t_load

        if args.benchmark:
            tracemalloc.stop()
            run_benchmark(index)
            return

        if args.mode == "boolean":
            t_q = time.perf_counter()
            if args.engine == "merge":
                doc_ids = boolean_search_merge(args.query, index)
            else:
                doc_ids = boolean_search_set(args.query, index)
            q_ms = (time.perf_counter() - t_q) * 1000
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(
                f"Found {len(doc_ids)} documents in {q_ms:.3f} ms "
                f"(load: {load_s:.3f} s, peak mem: {peak / (1024 * 1024):.2f} MB)"
            )
            for doc_id in doc_ids[: args.k]:
                meta = index.doc_meta.get(doc_id)
                label = f"{meta.title} ({meta.path})" if meta else f"doc_{doc_id}"
                print(f"  [{doc_id}] {label}")
            return

        engine = SearchEngine(index)
        t_q = time.perf_counter()
        results = engine.search(
            args.query, scorer=args.scorer, k=args.k, with_snippets=args.snippets
        )
        q_ms = (time.perf_counter() - t_q) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(
            f"Top {len(results)} results in {q_ms:.3f} ms "
            f"(scorer={args.scorer}, load={load_s:.3f} s, peak mem={peak / (1024 * 1024):.2f} MB)"
        )
        for res in results:
            print(f"  {res}")


if __name__ == "__main__":
    main(sys.argv[1:])
