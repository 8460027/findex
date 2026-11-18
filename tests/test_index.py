from pathlib import Path

from findex.index import DocMeta, Posting, build_index
from findex.search import evaluate_query
from findex.store import load_index, save_index


def test_dataclasses_hashable() -> None:
    p1 = Posting(doc_id=1, tf=3)
    p2 = Posting(doc_id=1, tf=3)
    p3 = Posting(doc_id=2, tf=1)
    assert p1 == p2
    assert hash(p1) == hash(p2)
    assert len({p1, p2, p3}) == 2

    m1 = DocMeta(doc_id=0, path="a.txt", title="A")
    m2 = DocMeta(doc_id=0, path="a.txt", title="A")
    assert len({m1, m2}) == 1


def test_index_and_boolean_search(tmp_path: Path) -> None:
    (tmp_path / "doc1.txt").write_text("Python search engine inverted index", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("Python dictionaries and hash tables", encoding="utf-8")
    (tmp_path / "doc3.txt").write_text("Inverted index compression algorithms", encoding="utf-8")

    idx = build_index(tmp_path, with_positions=True)
    assert len(idx.doc_meta) == 3

    # Перевірка сортування doc_id
    for plist in idx.postings.values():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(ids)

    for q in [
        "python",
        "python inverted",
        "python OR compression",
        "python NOT hash",
        "inverted index NOT compression",
    ]:
        res_merge = evaluate_query(idx, q, engine="merge")
        res_set = evaluate_query(idx, q, engine="set")
        assert res_merge == res_set

    assert evaluate_query(idx, "python inverted") == [0]
    assert evaluate_query(idx, "python OR compression") == [0, 1, 2]
    assert evaluate_query(idx, "python NOT hash") == [0]


def test_store_roundtrip(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world", encoding="utf-8")
    (corpus / "b.txt").write_text("hello python", encoding="utf-8")

    idx = build_index(corpus)

    for fmt, ext in [("pickle", "pkl"), ("json", "json"), ("binary", "bin")]:
        out = tmp_path / f"index.{ext}"
        save_index(idx, out, fmt=fmt)
        loaded = load_index(out, fmt=fmt)
        assert loaded.doc_meta == idx.doc_meta
        assert loaded.doc_lengths == idx.doc_lengths
        assert loaded.postings == idx.postings
