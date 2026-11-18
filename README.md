# findex — Search Engine in Python

Memory-efficient streaming corpus loader, Unicode tokenizer, inverted index, Boolean & phrase query parser, and TF-IDF / BM25 ranked search engine.

## Setup & Usage

```bash
uv sync
uv run pytest
uv run ruff check .
```

### Lab 01 — Corpus Stats
```powershell
$env:PYTHONPATH="src"
uv run python -m findex.stats data/ --limit 100
```

### Lab 02 — Build & Search Inverted Index
```powershell
$env:PYTHONPATH="src"

# Build and persist index (formats: pickle, json, binary)
uv run python -m findex.index data/ --out index.bin --format binary --positions

# Legacy Boolean search (engines: merge, set)
uv run python -m findex.search index.bin "python OR search" --format binary --mode boolean --engine merge
```

### Lab 03 — Ranked Search (BM25 / TF-IDF) & Snippets
```powershell
$env:PYTHONPATH="src"

# Ranked search with BM25, query AST parser, and highlighted snippets
uv run python -m findex.search index.bin "(python OR search) NOT java" --scorer bm25 --k 5 --snippets --verbose
```

---

## Lab 01 — Streams, Not Lists

### Tokenizer Policy
- **Normalization**: Unicode NFC followed by `.casefold()`.
- **Tokens**: Matches Unicode word characters, preserving internal apostrophes (`'` and `’`) and hyphens (`-`). Standalone underscores and punctuation are stripped.

---

## Lab 02 — The Inverted Index

### Architecture & Safety Notes
- **Data Models (`models.py`)**: `Posting` and `DocMeta` are implemented as `@dataclass(frozen=True, slots=True)`, making them immutable, compact, and hashable (usable in `set` and as `dict` keys).
- **Single-Pass Construction (`index.py`)**: Streams documents lazily via `iter_documents`, aggregates term frequencies per document with `Counter`, and appends sorted `Posting` records into `defaultdict(list)`.
- **Serialization & Security (`store.py`)**: Supports `pickle`, `json`, and a custom `binary` format (`uint32` arrays + JSON offset header). **Security warning:** `pickle.load()` in Python can execute arbitrary code via crafted opcodes; never load untrusted `.pkl` files from external sources.

### Memory Study (`tracemalloc`)

Run `uv run python -m findex.index data/ --memory-study` to reproduce:

| Postings representation | Peak memory (build) | Index file size | Load time |
|---|---|---|---|
| `list[Posting]` with plain `@dataclass` | 4.34 MB | 0.43 MB | 0.0273 s |
| `list[Posting]` with `slots=True` | 4.49 MB | 0.36 MB | 0.0544 s |
| `array('I')` pairs (doc_ids, tfs) | 2.27 MB | 0.23 MB | 0.0237 s |

*Optional positions cost (`--positions`):* Peak memory = **253.65 MB**, File size = **21.71 MB** (storing per-occurrence offset tuples multiplies pointer and tuple overhead).

**Where the bytes go:**
- **Plain `@dataclass`:** Every `Posting` allocates a full Python object header plus a dynamic per-instance `__dict__` (~100+ bytes overhead per posting) and two 28-byte `int` references.
- **`slots=True`:** Eliminates the per-instance `__dict__`, storing fields in fixed C-level struct slots and shrinking the serialized file (`0.36 MB` vs `0.43 MB`). (The slightly higher build peak `4.49 MB` reflects also holding `doc_lengths` and `doc_meta` in `InvertedIndex`).
- **`array('I')` pairs:** Eliminates per-posting Python objects entirely, packing `doc_id` and `tf` as contiguous 4-byte unsigned integers (`uint32`), cutting both RAM (`2.27 MB`) and disk footprint (`0.23 MB`) roughly in half.

### Serialization Formats Benchmark

Run `uv run python -m findex.store data/` to reproduce:

| Format | File size | Save time | Load time |
|---|---|---|---|
| `pickle` | 0.45 MB | 0.0508 s | 0.0663 s |
| `json` | 0.38 MB | 0.0608 s | 0.0392 s |
| `binary` | 0.31 MB | 0.0080 s | 0.0486 s |

### Boolean Retrieval: Two-Pointer Merge vs Set

Run `uv run python -m findex.search index.bin --benchmark` (2000 iterations per query):

- Common terms: `'document'` (`df=1500`), `'python'` (`df=1500`)
- Rare terms: `'1499'` (`df=1`), `'1498'` (`df=1`)

| Query type | `merge` time | `set` time | Speedup |
|---|---|---|---|
| Most common (AND) | 7691.92 ms | 520.12 ms | 0.07x |
| Most common (OR) | 8244.88 ms | 538.94 ms | 0.07x |
| Rarest (AND) | 80.15 ms | 96.17 ms | 1.20x |
| Rarest (OR) | 87.02 ms | 99.85 ms | 1.15x |

**Analysis:** Both algorithms run in $O(|A| + |B|)$ time. On long postings lists (`df=1500`), Python's built-in `set` intersection/union wins because its loop runs in C, whereas the two-pointer `merge` runs in interpreted Python bytecode. On short postings lists (`df=1`), two-pointer `merge` is **1.15x–1.20x faster** because it avoids hash table allocation and post-merge sorting overhead, while also streaming naturally without extra memory.

---

## Lab 03 — Ranking and the Object Model

### Python Data Model & Architecture
- **`Index` (`index.py`)**: Inherits `collections.abc.Mapping[str, list[Posting]]` and implements `__len__`, `__contains__`, `__getitem__`, `__iter__`, and `__repr__` (`Index(terms=1_513, docs=1_500)`). Exposes `num_docs` as a `@property` and `avg_doc_length` as a `@functools.cached_property` (`4802.00`).
- **Context Manager (`store.py`)**: `@contextmanager def open_index(path, fmt="binary")` ensures `index.close()` executes in `finally` even when exceptions occur.
- **Swappable `Scorer` Protocol (`ranking.py`)**: Structural `@runtime_checkable class Scorer(Protocol)` with `TfIdf` and `BM25(k1=1.5, b=0.75)` callable implementations; top-`k` results are extracted via `heapq.nlargest` into ordered `SearchResult` dataclasses.
- **Recursive-Descent Query Parser (`query.py`)**: Parses implicit `AND`, `OR`, `NOT`, parentheses, and positional `"exact phrases"` into composable AST nodes (`Term`, `Phrase`, `And`, `Or`, `Not`) overloading `&`, `|`, and `~`.
- **Decorators, Caching & Snippets (`decorators.py`, `search.py`, `snippets.py`)**: `@timed` (preserving metadata via `functools.wraps`) logs execution latency; `@functools.lru_cache(maxsize=256)` caches ranked query results; `build_snippet` extracts a ±80-character window around the first match and highlights terms as `[term]`.

### Ranking Sanity Checks (Corpus `data/`)

| Check | Observation on `data/` (`N=1500`, `avgdl=4802.00`) | Result |
|---|---|---|
| **1. Rare vs Common term** | Rare `'0'` (`df=1`) score = **6.9084** vs common `'швидко'` (`df=1500`) score = **0.0003** | Rare term outranks common term by >20,000× |
| **2. BM25 TF saturation** | `'швидко'`: `tf=1` → **0.0003**, `tf=19` → **0.0008**, `tf=20` → **0.0008** | 20th occurrence adds only **+0.37%** |
| **3. Document length normalization** | Synthetic corpus `data/` has uniform length (`len=4802` on both `#0` and `#1499` → **0.0003**); on variable-length docs (unit tests), short doc (`len=5`) outranks long doc (`len=81`) | Verified (`b=0.75` penalizes longer docs) |

### Caching & Snippet Log Demonstration

```text
INFO: search took 40.91 ms
INFO: search took 2.46 ms
Cache info: CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)
Top result: [0] doc_0000.txt (score=6.9084)
      Document [0] Python generators yield items lazily in constant memory. Пошуковий рушій працю...
```

### Precision@5 Evaluation (10 Queries: TF-IDF vs BM25)

*Note: For queries with 1 ground-truth document (`rel=1`), the maximum possible `P@5` is `1/5 = 0.20`; for `rel=2`, the maximum possible `P@5` is `2/5 = 0.40`. Both scorers placed all relevant documents at the top ranks.*

| Query | Relevant docs | TF-IDF P@5 | BM25 P@5 |
|---|---|---|---|
| `0 OR швидко` | 1 | 0.20 | 0.20 |
| `1 OR швидко` | 1 | 0.20 | 0.20 |
| `2 OR працює` | 1 | 0.20 | 0.20 |
| `756 швидко` | 1 | 0.20 | 0.20 |
| `757 рушій` | 1 | 0.20 | 0.20 |
| `758 NOT 0` | 1 | 0.20 | 0.20 |
| `(756 OR 757) швидко` | 2 | 0.40 | 0.40 |
| `756 OR рушій` | 1 | 0.20 | 0.20 |
| `757 OR працює` | 1 | 0.20 | 0.20 |
| `758 швидко` | 1 | 0.20 | 0.20 |
| **Mean P@5** | — | **0.22** | **0.22** |
