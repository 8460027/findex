# findex Р Р†Р вЂљРІР‚Сњ Search Engine in Python

> **Live Deployment (Lab 07):** [https://findex-search.onrender.com](https://findex-search.onrender.com) · Interactive OpenAPI: [/docs](https://findex-search.onrender.com/docs) · Health: [/health](https://findex-search.onrender.com/health)
[![CI](https://github.com/8460027/findex/actions/workflows/ci.yml/badge.svg)](https://github.com/8460027/findex/actions/workflows/ci.yml)

Memory-efficient streaming corpus loader, Unicode tokenizer, inverted index, Boolean & phrase query parser, TF-IDF / BM25 ranked search engine, and installable `typer` + `rich` CLI.

## Installation & Quick Start

> **Live Deployment (Lab 07):** [https://findex-search.onrender.com](https://findex-search.onrender.com) · Interactive OpenAPI: [/docs](https://findex-search.onrender.com/docs) · Health: [/health](https://findex-search.onrender.com/health)
### Install as a CLI tool (`uv`)
```bash
# From local repository
uv tool install .

# Or directly from a built wheel / GitHub Release asset
uv tool install dist/findex-0.7.0-py3-none-any.whl
```

### CLI Usage (`findex`)
```bash
# 1. Build and persist an inverted index with token positions
findex index data/ --out index.bin --positions

# 2. Ranked BM25 search with rich table output and highlighted snippets
findex search index.bin "(python OR search) NOT java" --scorer bm25 --k 5

# 3. Machine-readable JSON Lines output for piping
findex search index.bin "python" --scorer tfidf --k 5 --json

# 4. Inspect corpus or index statistics
findex stats data/ --limit 100
findex stats index.bin

# 5. Diagnostic logging on stderr (-v for INFO, -vv for DEBUG timings)
findex -vv search index.bin "python"
```

### Development & Quality Checks
```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -v --cov=findex --cov-report=term-missing
```

---

## Lab 04 Р Р†Р вЂљРІР‚Сњ Typing, Testing, Packaging, and CLI

> **Live Deployment (Lab 07):** [https://findex-search.onrender.com](https://findex-search.onrender.com) · Interactive OpenAPI: [/docs](https://findex-search.onrender.com/docs) · Health: [/health](https://findex-search.onrender.com/health)
### Static Typing (`pyright --strict`)
- Configured `[tool.pyright] typeCheckingMode = "strict"` in `pyproject.toml` with **0 errors, 0 warnings**.
- Structural `@runtime_checkable class Scorer(Protocol)` verified statically for both `BM25` and `TfIdf` without inheritance.
- Generator pipelines typed with `Iterator[Document]` / `Iterator[str]`; decorators typed with PEP 695 `ParamSpec` (`def timed[**P, R]`); structured dictionaries typed with `TypedDict` and `NamedTuple`.

### Test Suite & Coverage (`pytest` + `hypothesis`)
- **33 automated tests** (`100% passing`), including:
  - **5 Hypothesis property-based tests**: tokenizer non-empty/casefold invariants, tokenizer idempotence, two-pointer merge vs `set` oracle, binary/JSON/pickle index round-trip equality, and query AST render/parse round-trip.
  - **CLI integration tests** via `typer.testing.CliRunner` covering `index`, `search` (Rich table & `--json` lines), `stats`, `-vv` stderr logging, and nonzero exit codes on expected errors.
  - **Markers**: `@pytest.mark.slow` isolates larger corpus tests (skip locally with `pytest -m "not slow"`).
- **Code coverage (`pytest --cov=findex`): 76% overall**
  - High coverage on core production modules: `cli.py` (**95%**), `tokenize.py` (**100%**), `models.py` (**100%**), `decorators.py` (**100%**), `query.py` (**87%**), `ranking.py` (**85%**), `snippets.py` (**84%**), `store.py` (**83%**).
  - *Uncovered lines (24%)*: legacy Lab 01Р Р†Р вЂљРІР‚Сљ02 CLI `argparse` entry points and memory/speed benchmark helpers (`run_memory_study`, `run_benchmark`, `benchmark_formats`) in `index.py`, `search.py`, and `stats.py`, which are manual CLI profiling scripts rather than library/CLI runtime paths.

---

## Lab 01 Р Р†Р вЂљРІР‚Сњ Streams, Not Lists

> **Live Deployment (Lab 07):** [https://findex-search.onrender.com](https://findex-search.onrender.com) · Interactive OpenAPI: [/docs](https://findex-search.onrender.com/docs) · Health: [/health](https://findex-search.onrender.com/health)
### Tokenizer Policy
- **Normalization**: Unicode NFC followed by `.casefold()`.
- **Tokens**: Matches Unicode word characters, preserving internal apostrophes (`'` and `Р Р†Р вЂљРІвЂћСћ`) and hyphens (`-`). Standalone underscores and punctuation are stripped.

---

## Lab 02 Р Р†Р вЂљРІР‚Сњ The Inverted Index

> **Live Deployment (Lab 07):** [https://findex-search.onrender.com](https://findex-search.onrender.com) · Interactive OpenAPI: [/docs](https://findex-search.onrender.com/docs) · Health: [/health](https://findex-search.onrender.com/health)
### Architecture & Safety Notes
- **Data Models (`models.py`)**: `Posting` and `DocMeta` are implemented as `@dataclass(frozen=True, slots=True)`, making them immutable, compact, and hashable (usable in `set` and as `dict` keys).
- **Single-Pass Construction (`index.py`)**: Streams documents lazily via `iter_documents`, aggregates term frequencies per document with `Counter`, and appends sorted `Posting` records into `defaultdict(list)`.
- **Serialization & Security (`store.py`)**: Supports `pickle`, `json`, and a custom `binary` format (`uint32` arrays + JSON offset header). **Security warning:** `pickle.load()` in Python can execute arbitrary code via crafted opcodes; never load untrusted `.pkl` files from external sources.

### Memory Study (`tracemalloc`)

> **Live Deployment (Lab 07):** [https://findex-search.onrender.com](https://findex-search.onrender.com) · Interactive OpenAPI: [/docs](https://findex-search.onrender.com/docs) · Health: [/health](https://findex-search.onrender.com/health)
Run `uv run python -m findex.index data/ --memory-study` to reproduce:

| Postings representation | Peak memory (build) | Index file size | Load time |
|---|---|---|---|
| `list[Posting]` with plain `@dataclass` | 4.34 MB | 0.43 MB | 0.0273 s |
| `list[Posting]` with `slots=True` | 4.49 MB | 0.36 MB | 0.0544 s |
| `array('I')` pairs (doc_ids, tfs) | 2.27 MB | 0.23 MB | 0.0237 s |

*Optional positions cost (`--positions`):* Peak memory = **253.65 MB**, File size = **21.71 MB** (storing per-occurrence offset tuples multiplies pointer and tuple overhead).

### Serialization Formats Benchmark

> **Live Deployment (Lab 07):** [https://findex-search.onrender.com](https://findex-search.onrender.com) · Interactive OpenAPI: [/docs](https://findex-search.onrender.com/docs) · Health: [/health](https://findex-search.onrender.com/health)
Run `uv run python -m findex.store data/` to reproduce:

| Format | File size | Save time | Load time |
|---|---|---|---|
| `pickle` | 0.45 MB | 0.0508 s | 0.0663 s |
| `json` | 0.38 MB | 0.0608 s | 0.0392 s |
| `binary` | 0.31 MB | 0.0080 s | 0.0486 s |

### Boolean Retrieval: Two-Pointer Merge vs Set

> **Live Deployment (Lab 07):** [https://findex-search.onrender.com](https://findex-search.onrender.com) · Interactive OpenAPI: [/docs](https://findex-search.onrender.com/docs) · Health: [/health](https://findex-search.onrender.com/health)
Run `uv run python -m findex.search index.bin --benchmark` (2000 iterations per query):

- Common terms: `'document'` (`df=1500`), `'python'` (`df=1500`)
- Rare terms: `'1499'` (`df=1`), `'1498'` (`df=1`)

| Query type | `merge` time | `set` time | Speedup |
|---|---|---|---|
| Most common (AND) | 7691.92 ms | 520.12 ms | 0.07x |
| Most common (OR) | 8244.88 ms | 538.94 ms | 0.07x |
| Rarest (AND) | 80.15 ms | 96.17 ms | 1.20x |
| Rarest (OR) | 87.02 ms | 99.85 ms | 1.15x |

---

## Lab 03 Р Р†Р вЂљРІР‚Сњ Ranking and the Object Model

> **Live Deployment (Lab 07):** [https://findex-search.onrender.com](https://findex-search.onrender.com) · Interactive OpenAPI: [/docs](https://findex-search.onrender.com/docs) · Health: [/health](https://findex-search.onrender.com/health)
### Python Data Model & Architecture
- **`Index` (`index.py`)**: Inherits `collections.abc.Mapping[str, list[Posting]]` and implements `__len__`, `__contains__`, `__getitem__`, `__iter__`, and `__repr__` (`Index(terms=1_513, docs=1_500)`). Exposes `num_docs` as a `@property` and `avg_doc_length` as a `@functools.cached_property` (`4802.00`).
- **Context Manager (`store.py`)**: `@contextmanager def open_index(path, fmt="binary")` ensures `index.close()` executes in `finally` even when exceptions occur.
- **Swappable `Scorer` Protocol (`ranking.py`)**: Structural `@runtime_checkable class Scorer(Protocol)` with `TfIdf` and `BM25(k1=1.5, b=0.75)` callable implementations; top-`k` results are extracted via `heapq.nlargest` into ordered `SearchResult` dataclasses.
- **Recursive-Descent Query Parser (`query.py`)**: Parses implicit `AND`, `OR`, `NOT`, parentheses, and positional `"exact phrases"` into composable AST nodes (`Term`, `Phrase`, `And`, `Or`, `Not`) overloading `&`, `|`, and `~`.
- **Decorators, Caching & Snippets (`decorators.py`, `search.py`, `snippets.py`)**: `@timed` (preserving metadata via `functools.wraps`) logs execution latency at `DEBUG`; `@functools.lru_cache(maxsize=256)` caches ranked query results; `build_snippet` extracts a Р вЂ™Р’В±80-character window around the first match and highlights terms as `[term]`.

### Ranking Sanity Checks (Corpus `data/`)

> **Live Deployment (Lab 07):** [https://findex-search.onrender.com](https://findex-search.onrender.com) · Interactive OpenAPI: [/docs](https://findex-search.onrender.com/docs) · Health: [/health](https://findex-search.onrender.com/health)
| Check | Observation on `data/` (`N=1500`, `avgdl=4802.00`) | Result |
|---|---|---|
| **1. Rare vs Common term** | Rare `'0'` (`df=1`) score = **6.9084** vs common `'Р РЋРІвЂљВ¬Р В Р вЂ Р В РЎвЂР В РўвЂР В РЎвЂќР В РЎвЂў'` (`df=1500`) score = **0.0003** | Rare term outranks common term by >20,000Р вЂњРІР‚вЂќ |
| **2. BM25 TF saturation** | `'Р РЋРІвЂљВ¬Р В Р вЂ Р В РЎвЂР В РўвЂР В РЎвЂќР В РЎвЂў'`: `tf=1` Р Р†РІР‚В РІР‚в„ў **0.0003**, `tf=19` Р Р†РІР‚В РІР‚в„ў **0.0008**, `tf=20` Р Р†РІР‚В РІР‚в„ў **0.0008** | 20th occurrence adds only **+0.37%** |
| **3. Document length normalization** | Synthetic corpus `data/` has uniform length (`len=4802` on both `#0` and `#1499` Р Р†РІР‚В РІР‚в„ў **0.0003**); on variable-length docs (unit tests), short doc (`len=5`) outranks long doc (`len=81`) | Verified (`b=0.75` penalizes longer docs) |

### Caching & Snippet Log Demonstration

> **Live Deployment (Lab 07):** [https://findex-search.onrender.com](https://findex-search.onrender.com) · Interactive OpenAPI: [/docs](https://findex-search.onrender.com/docs) · Health: [/health](https://findex-search.onrender.com/health)
```text
INFO: search took 40.91 ms
INFO: search took 2.46 ms
Cache info: CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)
Top result: [0] doc_0000.txt (score=6.9084)
      Document [0] Python generators yield items lazily in constant memory. Р В РЎСџР В РЎвЂўР РЋРІвЂљВ¬Р РЋРЎвЂњР В РЎвЂќР В РЎвЂўР В Р вЂ Р В РЎвЂР В РІвЂћвЂ“ Р РЋР вЂљР РЋРЎвЂњР РЋРІвЂљВ¬Р РЋРІР‚вЂњР В РІвЂћвЂ“ Р В РЎвЂ”Р РЋР вЂљР В Р’В°Р РЋРІР‚В Р РЋР вЂ№...
```

### Precision@5 Evaluation (10 Queries: TF-IDF vs BM25)

> **Live Deployment (Lab 07):** [https://findex-search.onrender.com](https://findex-search.onrender.com) · Interactive OpenAPI: [/docs](https://findex-search.onrender.com/docs) · Health: [/health](https://findex-search.onrender.com/health)
| Query | Relevant docs | TF-IDF P@5 | BM25 P@5 |
|---|---|---|---|
| `0 OR Р РЋРІвЂљВ¬Р В Р вЂ Р В РЎвЂР В РўвЂР В РЎвЂќР В РЎвЂў` | 1 | 0.20 | 0.20 |
| `1 OR Р РЋРІвЂљВ¬Р В Р вЂ Р В РЎвЂР В РўвЂР В РЎвЂќР В РЎвЂў` | 1 | 0.20 | 0.20 |
| `2 OR Р В РЎвЂ”Р РЋР вЂљР В Р’В°Р РЋРІР‚В Р РЋР вЂ№Р РЋРІР‚Сњ` | 1 | 0.20 | 0.20 |
| `756 Р РЋРІвЂљВ¬Р В Р вЂ Р В РЎвЂР В РўвЂР В РЎвЂќР В РЎвЂў` | 1 | 0.20 | 0.20 |
| `757 Р РЋР вЂљР РЋРЎвЂњР РЋРІвЂљВ¬Р РЋРІР‚вЂњР В РІвЂћвЂ“` | 1 | 0.20 | 0.20 |
| `758 NOT 0` | 1 | 0.20 | 0.20 |
| `(756 OR 757) Р РЋРІвЂљВ¬Р В Р вЂ Р В РЎвЂР В РўвЂР В РЎвЂќР В РЎвЂў` | 2 | 0.40 | 0.40 |
| `756 OR Р РЋР вЂљР РЋРЎвЂњР РЋРІвЂљВ¬Р РЋРІР‚вЂњР В РІвЂћвЂ“` | 1 | 0.20 | 0.20 |
| `757 OR Р В РЎвЂ”Р РЋР вЂљР В Р’В°Р РЋРІР‚В Р РЋР вЂ№Р РЋРІР‚Сњ` | 1 | 0.20 | 0.20 |
| `758 Р РЋРІвЂљВ¬Р В Р вЂ Р В РЎвЂР В РўвЂР В РЎвЂќР В РЎвЂў` | 1 | 0.20 | 0.20 |
| **Mean P@5** | Р Р†Р вЂљРІР‚Сњ | **0.22** | **0.22** |