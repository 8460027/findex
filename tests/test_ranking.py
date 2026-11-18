from __future__ import annotations

import logging

import pytest

from findex.decorators import timed
from findex.index import build_index
from findex.models import Posting
from findex.query import And, Not, Or, Phrase, Term, parse_query
from findex.ranking import BM25, Scorer, SearchResult, TfIdf, rank_documents
from findex.search import SearchEngine
from findex.store import open_index, save_index


def _make_sample_corpus(tmp_path):
    docs = {
        "doc0.txt": "python search engine builds inverted index fast",
        "doc1.txt": "python python python python python python python python python python "
        "python python python python python python python python python python",
        "doc2.txt": "rare unicorn algorithm in python",
        "doc3.txt": "python " + " ".join(["filler"] * 80),
    }
    for name, content in docs.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    return tmp_path


def test_index_dunders_and_properties(tmp_path):
    corpus = _make_sample_corpus(tmp_path)
    ix = build_index(corpus, with_positions=True)

    assert len(ix) > 0
    assert "python" in ix
    assert "nonexistent_term_xyz" not in ix
    assert len(ix["python"]) == 4

    with pytest.raises(KeyError):
        _ = ix["nonexistent_term_xyz"]

    assert set(ix) == set(ix.postings.keys())
    assert repr(ix) == f"Index(terms={len(ix):_}, docs=4)"
    assert ix.num_docs == 4
    assert ix.df("unicorn") == 1
    assert ix.df("nonexistent_term_xyz") == 0
    assert ix.avg_doc_length == sum(ix.doc_lengths.values()) / 4


def test_open_index_context_manager_cleanup(tmp_path):
    corpus = _make_sample_corpus(tmp_path)
    ix = build_index(corpus)
    out = tmp_path / "test.bin"
    save_index(ix, out, fmt="binary")

    with open_index(out, fmt="binary") as loaded:
        assert not loaded.closed
        assert "python" in loaded
    assert loaded.closed

    with pytest.raises(RuntimeError), open_index(out, fmt="binary") as loaded_err:
        assert not loaded_err.closed
        raise RuntimeError("boom")
    assert loaded_err.closed


def test_scorer_protocol_and_sanity_checks(tmp_path):
    corpus = _make_sample_corpus(tmp_path)
    ix = build_index(corpus)

    assert isinstance(BM25(), Scorer)
    assert isinstance(TfIdf(), Scorer)

    # Sanity check 1: rare term ("unicorn", df=1) outranks common term ("python", df=4)
    bm25 = BM25()
    p_rare = Posting(doc_id=2, tf=1)
    p_common = Posting(doc_id=2, tf=1)
    assert bm25.score("unicorn", p_rare, ix) > bm25.score("python", p_common, ix)

    # Sanity check 2: BM25 saturation — 20th repetition adds almost nothing (< 5% over 19th)
    s19 = bm25.score("python", Posting(doc_id=1, tf=19), ix)
    s20 = bm25.score("python", Posting(doc_id=1, tf=20), ix)
    assert (s20 - s19) / s19 < 0.05

    # Sanity check 3: short doc (doc2, len=5) outranks long doc (doc3, len=81) on 1 match
    s_short = bm25.score("python", Posting(doc_id=2, tf=1), ix)
    s_long = bm25.score("python", Posting(doc_id=3, tf=1), ix)
    assert s_short > s_long

    # Swappable scorers produce ordered SearchResult items
    r_bm25 = rank_documents(["unicorn", "python"], ix, scorer=BM25(), k=2)
    r_tfidf = rank_documents(["unicorn", "python"], ix, scorer=TfIdf(), k=2)
    assert r_bm25[0][0] == 2
    assert r_tfidf[0][0] == 2

    res_a = SearchResult(score=1.2, doc_id=1, title="a")
    res_b = SearchResult(score=2.5, doc_id=2, title="b")
    assert sorted([res_b, res_a]) == [res_a, res_b]


def test_query_parser_trees_and_phrases(tmp_path):
    # Tree equality & operator precedence: implicit AND binds tighter than OR
    assert parse_query("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))
    assert parse_query("(a OR b) AND NOT c") == And(Or(Term("a"), Term("b")), Not(Term("c")))
    assert parse_query('"search engine"') == Phrase(("search", "engine"))

    # Operator overloading on QueryNode objects
    prog_tree = Term("python") & (Term("unicorn") | ~Term("filler"))
    assert isinstance(prog_tree, And)

    corpus = _make_sample_corpus(tmp_path)
    ix = build_index(corpus, with_positions=True)

    assert parse_query('"search engine"').evaluate(ix) == {0}
    assert parse_query('"engine search"').evaluate(ix) == set()
    assert parse_query("python NOT filler").evaluate(ix) == {0, 1, 2}


def test_timed_cache_and_snippets(tmp_path, caplog):
    @timed
    def sample_fn(x: int) -> int:
        """Sample docstring."""
        return x * 2

    assert sample_fn.__name__ == "sample_fn"
    assert sample_fn.__doc__ == "Sample docstring."

    corpus = _make_sample_corpus(tmp_path)
    ix = build_index(corpus, with_positions=True)
    engine = SearchEngine(ix)

    with caplog.at_level(logging.INFO, logger="findex"):
        r1 = engine.search("unicorn", scorer="bm25", with_snippets=True)
        r2 = engine.search("unicorn", scorer="bm25", with_snippets=True)

    assert r1 == r2
    assert len(r1) == 1
    assert "[unicorn]" in r1[0].snippet
    info = engine.cache_info()
    assert info.hits == 1
    assert info.misses == 1
