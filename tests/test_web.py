from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.index import Index, build_index
from findex.search import SearchEngine
from findex.web.app import create_app, get_engine, get_index, get_optional_index
from findex.web.config import Settings


@pytest.fixture
def tiny_index(tmp_path: Path) -> Index:
    corpus = tmp_path / "tiny.jsonl"
    docs = [
        {
            "id": "doc-0",
            "title": "Asyncio Event Loop",
            "text": "Cooperative multitasking in Python uses asyncio coroutines and await.",
        },
        {
            "id": "doc-1",
            "title": "FastAPI Web Services",
            "text": "FastAPI validates query parameters using Pydantic and type hints.",
        },
        {
            "id": "doc-2",
            "title": "BM25 Ranking Model",
            "text": "Okapi BM25 ranks documents by term frequency and inverse document frequency.",
        },
    ]
    corpus.write_text("\n".join(json.dumps(d) for d in docs) + "\n", encoding="utf-8")
    return build_index(corpus, executor="serial")


def test_api_routes_with_dependency_overrides(tiny_index: Index) -> None:
    app = create_app(Settings(index_path="nonexistent_ignored_by_override.jsonl"))
    engine = SearchEngine(tiny_index)
    app.dependency_overrides[get_optional_index] = lambda: tiny_index
    app.dependency_overrides[get_index] = lambda: tiny_index
    app.dependency_overrides[get_engine] = lambda: engine

    client = TestClient(app)

    # /health -> 200
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"
    assert h.json()["doc_count"] == 3
    assert "X-Request-ID" in h.headers

    # /stats -> 200
    st = client.get("/stats")
    assert st.status_code == 200
    st_data = st.json()
    assert st_data["doc_count"] == 3
    assert st_data["vocabulary_size"] > 0

    # /search -> 200 with pagination and scorer
    sr = client.get(
        "/search", params={"q": "asyncio OR fastapi", "k": 1, "page": 1, "scorer": "bm25"}
    )
    assert sr.status_code == 200
    payload = sr.json()
    assert payload["total"] == 2
    assert payload["total_pages"] == 2
    assert payload["page"] == 1
    assert len(payload["results"]) == 1
    assert payload["results"][0]["snippet"] != ""

    # Page 2
    sr2 = client.get(
        "/search", params={"q": "asyncio OR fastapi", "k": 1, "page": 2, "scorer": "tfidf"}
    )
    assert sr2.status_code == 200
    assert len(sr2.json()["results"]) == 1

    # /docs/{doc_id} -> 200 and 404
    doc_ok = client.get("/docs/0")
    assert doc_ok.status_code == 200
    assert doc_ok.json()["title"] == "doc-0"
    assert "coroutines" in doc_ok.json()["text"]

    doc_404 = client.get("/docs/999")
    assert doc_404.status_code == 404
    assert "detail" in doc_404.json()


def test_validation_422_and_openapi_docs(tiny_index: Index) -> None:
    app = create_app()
    app.dependency_overrides[get_optional_index] = lambda: tiny_index
    app.dependency_overrides[get_index] = lambda: tiny_index
    app.dependency_overrides[get_engine] = lambda: SearchEngine(tiny_index)
    client = TestClient(app)

    # Invalid integer parameter k=abc -> 422
    bad_k = client.get("/search", params={"q": "python", "k": "abc"})
    assert bad_k.status_code == 422
    assert "detail" in bad_k.json()

    # Invalid scorer literal -> 422
    bad_scorer = client.get("/search", params={"q": "python", "scorer": "unknown"})
    assert bad_scorer.status_code == 422

    # Missing required q -> 422
    missing_q = client.get("/search")
    assert missing_q.status_code == 422

    # Invalid Boolean query syntax -> 422
    bad_syntax = client.get("/search", params={"q": "AND OR ("})
    assert bad_syntax.status_code == 422

    # OpenAPI schema and /docs UI
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/docs").status_code == 200


def test_health_returns_503_when_index_unloaded() -> None:
    app = create_app()
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 503
    assert res.json()["status"] == "unavailable"


def test_lifespan_fails_fast_on_missing_index_path() -> None:
    app = create_app(Settings(index_path="does_not_exist_12345.bin"))
    with pytest.raises(FileNotFoundError, match="INDEX_PATH does not exist"), TestClient(app):
        pass


def test_jinja_ui_routes(tiny_index: Index) -> None:
    app = create_app()
    app.dependency_overrides[get_optional_index] = lambda: tiny_index
    app.dependency_overrides[get_index] = lambda: tiny_index
    app.dependency_overrides[get_engine] = lambda: SearchEngine(tiny_index)
    client = TestClient(app)

    home = client.get("/", params={"q": "asyncio", "scorer": "bm25"})
    assert home.status_code == 200
    assert "<mark>asyncio</mark>" in home.text.lower()
    assert "doc-0" in home.text

    doc_page = client.get("/doc/1")
    assert doc_page.status_code == 200
    assert "doc-1" in doc_page.text

    css = client.get("/static/style.css")
    assert css.status_code == 200
    assert "--primary" in css.text
