# Lab 08 — Baseline Profiling Report (M1)

Baseline profiling was conducted **before any NumPy optimizations** using `cProfile` on a 5,000-document JSONL corpus (250 uncached 3+ term BM25 queries = 3,750,000 posting evaluations). Visual call-stack breakdown is saved in [`docs/search_baseline.svg`](search_baseline.svg).

## Top 3 Hotspots (Baseline)

| Rank | Function | Calls | Self Time | Cumulative Time | Execution Type | Hypothesis |
|---|---|---:|---:|---:|---|---|
| 1 | `ranking.py:61 (BM25.score)` | 3,750,000 | 5.690 s | 11.855 s (73.0%) | Pure Python bytecode | Recomputes `math.log` for `idf` on every single posting, performs Python float boxing, and triggers per-posting method dispatch overhead. |
| 2 | `ranking.py:84 (rank_documents)` | 250 | 2.660 s | 15.497 s (95.4%) | Pure Python bytecode | Iterates over 3.75M `Posting` dataclasses in a Python `for` loop, performing `scores.get()` and dictionary hash-table insertions per match. |
| 3 | `index.py:76,86,89 (num_docs / doc_length / df)` | 15,000,002 | 3.771 s | 5.594 s (34.4%) | Pure Python property/dict lookup | Repeated property lookups, `len()` calls (11.25M), and dict lookups inside the inner scoring loop instead of contiguous `int32` array indexing. |

## Optimization Plan (M2)
1. Store per-term postings as contiguous NumPy `int32` arrays (`doc_ids`, `tfs`) and document lengths as a single contiguous `int32[N]` array.
2. Vectorize `BM25` and `TfIdf` using NumPy ufuncs (`np.log`), fancy indexing (`doc_lengths[doc_ids]`), and scatter-add (`np.add.at` / buffered slice accumulation).
3. Replace full heap/sort extraction with `np.argpartition` for $O(N)$ top-$k$ selection while preserving identical ranking order to the Lab 3 scalar reference implementation.