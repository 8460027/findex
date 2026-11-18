# findex — Streams, Not Lists (Lab 01)

Memory-efficient streaming corpus loader and Unicode-normalized tokenizer built with Python generators.

## Setup & Usage

    uv sync
    uv run pytest
    uv run ruff check .
    PYTHONPATH=src uv run python -m findex.stats data/ --limit 100

## Tokenizer Policy

- **Normalization**: Unicode NFC followed by .casefold().
- **Tokens**: Matches Unicode word characters, preserving internal apostrophes (' and ’) for contractions and Ukrainian words (don't, п'ять) as well as internal hyphens (-) for compound words (state-of-the-art, південно-західний). Standalone underscores and punctuation are stripped.

## Benchmark (eager vs lazy)

Run `PYTHONPATH=src uv run python -m findex.stats data/ --benchmark` to reproduce:

| Version | Documents | Peak memory | Elapsed |
|---|---|---|---|
| eager (lists) | 1500 | <ВСТАВТЕ_MB> MB | <ВСТАВТЕ_S> s |
| lazy (generators) | 1500 | <ВСТАВТЕ_MB> MB | <ВСТАВТЕ_S> s |
