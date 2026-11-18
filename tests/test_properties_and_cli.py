from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from findex.cli import app
from findex.index import Index, build_index
from findex.models import DocMeta, Posting
from findex.query import And, Not, Or, Phrase, QueryNode, Term, parse_query
from findex.ranking import BM25, TfIdf
from findex.search import (
    SearchEngine,
    boolean_search_merge,
    boolean_search_set,
    intersect_postings,
    union_postings,
)
from findex.stats import run_eager_stats, run_lazy_stats
from findex.store import FormatType, load_index, save_index

runner = CliRunner()

RESERVED_WORDS = {"and", "or", "not"}
word_strategy = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8).filter(
    lambda w: w not in RESERVED_WORDS
)


def _render_query(node: QueryNode) -> str:
    if isinstance(node, Term):
        return node.value
    if isinstance(node, Phrase):
        return '"' + " ".join(node.terms) + '"'
    if isinstance(node, Not):
        return f"NOT ({_render_query(node.operand)})"
    if isinstance(node, And):
        return f"({_render_query(node.left)} AND {_render_query(node.right)})"
    if isinstance(node, Or):
        return f"({_render_query(node.left)} OR {_render_query(node.right)})"
    raise TypeError(f"Unsupported node: {node!r}")


# --- Hypothesis Property-Based Tests (5 properties total across suite) ---


@given(
    st.lists(st.integers(min_value=0, max_value=500), unique=True).map(sorted),
    st.lists(st.integers(min_value=0, max_value=500), unique=True).map(sorted),
)
def test_prop_merge_matches_set_oracle(ids_a: list[int], ids_b: list[int]) -> None:
    postings_a = [Posting(doc_id=d, tf=1) for d in ids_a]
    postings_b = [Posting(doc_id=d, tf=1) for d in ids_b]

    assert intersect_postings(postings_a, postings_b) == sorted(set(ids_a) & set(ids_b))
    assert union_postings(postings_a, postings_b) == sorted(set(ids_a) | set(ids_b))


@given(
    docs=st.dictionaries(
        keys=word_strategy,
        values=st.lists(word_strategy, min_size=1, max_size=12),
        min_size=1,
        max_size=6,
    ),
    fmt=st.sampled_from(["binary", "json", "pickle"]),
)
@settings(max_examples=25, deadline=None)
def test_prop_index_roundtrip_all_formats(
    docs: dict[str, list[str]], fmt: FormatType, tmp_path_factory: pytest.TempPathFactory
) -> None:
    tmp_dir = tmp_path_factory.mktemp("roundtrip")
    for idx, (name, words) in enumerate(docs.items()):
        (tmp_dir / f"{idx}_{name}.txt").write_text(" ".join(words), encoding="utf-8")

    ix = build_index(tmp_dir, with_positions=True)
    out = tmp_dir / f"index.{fmt}"
    save_index(ix, out, fmt=fmt)
    restored = load_index(out, fmt=fmt)

    assert dict(restored.postings) == dict(ix.postings)
    assert restored.doc_lengths == ix.doc_lengths
    assert restored.doc_meta == ix.doc_meta


@given(
    tree=st.recursive(
        word_strategy.map(Term),
        lambda children: st.one_of(
            st.tuples(children, children).map(lambda pair: And(pair[0], pair[1])),
            st.tuples(children, children).map(lambda pair: Or(pair[0], pair[1])),
            children.map(Not),
        ),
        max_leaves=6,
    )
)
def test_prop_query_parser_roundtrip(tree: QueryNode) -> None:
    rendered = _render_query(tree)
    reparsed = parse_query(rendered)
    assert reparsed == tree


# --- CLI Tests (typer.testing.CliRunner) ---


def test_cli_help() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "index" in res.stdout
    assert "search" in res.stdout
    assert "stats" in res.stdout


def test_cli_index_and_search_rich_table(sample_corpus_dir: Path, tmp_path: Path) -> None:
    out_idx = tmp_path / "cli_idx.bin"
    res_idx = runner.invoke(
        app,
        ["-vv", "index", str(sample_corpus_dir), "--out", str(out_idx), "--positions"],
    )
    assert res_idx.exit_code == 0
    assert out_idx.exists()
    assert "Indexed" in res_idx.stdout

    res_search = runner.invoke(
        app,
        ["search", str(out_idx), '"inverted index" OR unicorn', "--scorer", "bm25", "--k", "3"],
    )
    assert res_search.exit_code == 0
    assert "Search Results" in res_search.stdout


def test_cli_search_json_lines_output(persisted_index_path: Path) -> None:
    res = runner.invoke(
        app,
        ["search", str(persisted_index_path), "python", "--scorer", "tfidf", "--k", "2", "--json"],
    )
    assert res.exit_code == 0
    lines = [line for line in res.stdout.strip().splitlines() if line.strip()]
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert {"rank", "doc_id", "title", "score", "snippet"} <= set(obj.keys())
        assert isinstance(obj["score"], float)


def test_cli_search_no_matches(persisted_index_path: Path) -> None:
    res = runner.invoke(app, ["search", str(persisted_index_path), "nonexistenttermxyz"])
    assert res.exit_code == 0
    assert "No documents matched" in res.stdout


def test_cli_stats_corpus_and_index(sample_corpus_dir: Path, persisted_index_path: Path) -> None:
    res_corpus = runner.invoke(app, ["stats", str(sample_corpus_dir), "--top", "5"])
    assert res_corpus.exit_code == 0
    assert "Corpus Statistics" in res_corpus.stdout

    res_index = runner.invoke(app, ["stats", str(persisted_index_path), "--top", "5"])
    assert res_index.exit_code == 0
    assert "Index Statistics" in res_index.stdout


def test_cli_error_exit_codes(tmp_path: Path, persisted_index_path: Path) -> None:
    missing = tmp_path / "does_not_exist.bin"
    res_missing_corpus = runner.invoke(app, ["index", str(missing)])
    assert res_missing_corpus.exit_code == 1

    res_missing_idx = runner.invoke(app, ["search", str(missing), "python"])
    assert res_missing_idx.exit_code == 1

    res_bad_query = runner.invoke(app, ["search", str(persisted_index_path), "(python OR"])
    assert res_bad_query.exit_code == 1

    res_missing_stats = runner.invoke(app, ["stats", str(missing)])
    assert res_missing_stats.exit_code == 1


# --- Additional Core Coverage Tests ---


def test_eager_and_lazy_stats_agree(sample_corpus_dir: Path) -> None:
    lazy = run_lazy_stats(sample_corpus_dir)
    eager = run_eager_stats(sample_corpus_dir)
    assert lazy["doc_count"] == eager["doc_count"] == 6
    assert lazy["total_tokens"] == eager["total_tokens"]
    assert lazy["vocab_size"] == eager["vocab_size"]


def test_boolean_merge_and_set_consistency(sample_index: Index) -> None:
    for q in ["python AND search", "python OR unicorn", "nonexistent"]:
        assert boolean_search_merge(q, sample_index) == boolean_search_set(q, sample_index)


def test_search_engine_scorers_and_invalid_scorer(sample_index: Index) -> None:
    engine = SearchEngine(sample_index)
    bm25_res = engine.search("python", scorer=BM25(), k=3)
    tfidf_res = engine.search("python", scorer=TfIdf(), k=3)
    assert len(bm25_res) == 3
    assert len(tfidf_res) == 3
    with pytest.raises(TypeError):
        engine.search("python", scorer="unknown")


def test_binary_index_validation_errors(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.bin"
    bad_file.write_bytes(b"BADMAGIC12345678")
    with pytest.raises(ValueError, match="Invalid index file magic"):
        load_index(bad_file, fmt="binary")


@pytest.mark.slow
def test_slow_large_synthetic_corpus_pipeline(tmp_path: Path) -> None:
    """End-to-end check on 400 synthetic documents (deselectable via -m 'not slow')."""
    corpus = tmp_path / "big"
    corpus.mkdir()
    for i in range(400):
        (corpus / f"d_{i:03d}.txt").write_text(
            f"Document {i} about python search ranking and inverted index structures.",
            encoding="utf-8",
        )
    ix = build_index(corpus)
    assert ix.num_docs == 400
    assert isinstance(ix.doc_meta[0], DocMeta)
    assert isinstance(ix, Index)
