from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.crawler.crawl import canonicalize_url, crawl
from findex.crawler.fetch import USER_AGENT, fetch
from findex.crawler.robots import RobotsCache
from findex.index import build_index
from findex.search import SearchEngine

runner = CliRunner()


def test_canonicalize_url_normalizes_components() -> None:
    assert (
        canonicalize_url("HTTP://Example.COM:80/docs/guide/#section?b=2&a=1")
        == "http://example.com/docs/guide"
    )
    assert (
        canonicalize_url("/api/ref/?z=9&a=1#top", base_url="https://example.com:443/docs/")
        == "https://example.com/api/ref?a=1&z=9"
    )
    assert canonicalize_url("mailto:test@example.com") is None


def test_fetch_retries_429_with_retry_after_and_503_but_not_404() -> None:
    calls: Counter[str] = Counter()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["User-Agent"] == USER_AGENT
        path = request.url.path
        calls[path] += 1
        if path == "/rate-limited":
            if calls[path] == 1:
                return httpx.Response(429, headers={"Retry-After": "0.02"}, text="slow down")
            return httpx.Response(200, text="<html><title>OK</title><body>Recovered</body></html>")
        if path == "/flaky":
            if calls[path] < 3:
                return httpx.Response(503, text="unavailable")
            return httpx.Response(200, text="<html><body>Back online</body></html>")
        return httpx.Response(404, text="not found")

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, headers={"User-Agent": USER_AGENT}
        ) as client:
            r429 = await fetch(client, "https://example.com/rate-limited", base_backoff=0.01)
            assert r429.status == 200
            assert r429.attempts == 2
            assert "Recovered" in r429.html

            r503 = await fetch(client, "https://example.com/flaky", base_backoff=0.01)
            assert r503.status == 200
            assert r503.attempts == 3

            r404 = await fetch(client, "https://example.com/missing", base_backoff=0.01)
            assert r404.status == 404
            assert r404.attempts == 1
            assert calls["/missing"] == 1

    asyncio.run(run())


def test_fetch_timeout_surfaces_error_after_retries() -> None:
    async def slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return httpx.Response(200, text="too late")

    async def run() -> None:
        transport = httpx.MockTransport(slow_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            res = await fetch(
                client,
                "https://example.com/hang",
                timeout=0.03,
                max_retries=1,
                base_backoff=0.01,
            )
            assert res.status == 0
            assert res.attempts == 2
            assert res.error is not None and "TimeoutError" in res.error

    asyncio.run(run())


def test_robots_cache_fetches_once_and_enforces_disallow() -> None:
    robots_hits = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal robots_hits
        if request.url.path == "/robots.txt":
            robots_hits += 1
            return httpx.Response(
                200,
                text="User-agent: *\nDisallow: /private/\nAllow: /\n",
            )
        return httpx.Response(404)

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            cache = RobotsCache(client)
            results = await asyncio.gather(
                cache.allowed("https://docs.example.org/guide/intro"),
                cache.allowed("https://docs.example.org/private/secret"),
                cache.allowed("https://docs.example.org/api/v1"),
            )
            assert results == [True, False, True]
            assert robots_hits == 1

    asyncio.run(run())


def test_crawl_respects_robots_domains_dedup_max_pages_and_feeds_index(tmp_path: Path) -> None:
    site: dict[str, str] = {
        "/robots.txt": "User-agent: *\nDisallow: /secret\n",
        "/": (
            "<html><head><title>Home</title></head><body>"
            "Welcome to async search engine. "
            "<a href='/p1#top'>P1</a> "
            "<a href='/p1/'>P1 dup</a> "
            "<a href='/p2'>P2</a> "
            "<a href='/secret'>Secret</a> "
            "<a href='https://evil.org/external'>External</a>"
            "</body></html>"
        ),
        "/p1": (
            "<html><head><title>Page 1</title></head><body>"
            "Coroutines yield to the event loop. <a href='/p3'>P3</a>"
            "</body></html>"
        ),
        "/p2": (
            "<html><head><title>Page 2</title></head><body>"
            "Structured concurrency with TaskGroup. <a href='/'>Home</a>"
            "</body></html>"
        ),
        "/p3": (
            "<html><head><title>Page 3</title></head><body>"
            "Semaphores bound per-host concurrency."
            "</body></html>"
        ),
        "/secret": "<html><body>Should never be fetched</body></html>",
    }
    fetched_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != "mirror.local":
            pytest.fail(f"Unexpected external host request: {request.url}")
        path = request.url.path
        fetched_paths.append(path)
        if path in site:
            return httpx.Response(200, text=site[path])
        return httpx.Response(404, text="not found")

    async def run() -> list[dict[str, str]]:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport, headers={"User-Agent": USER_AGENT}
        ) as client:
            pages = [
                p
                async for p in crawl(
                    ["http://mirror.local/"],
                    max_pages=3,
                    concurrency=4,
                    per_host=2,
                    per_host_delay=0.0,
                    client=client,
                )
            ]
        return [
            {
                "id": p.url,
                "url": p.url,
                "title": p.title,
                "text": p.text,
                "fetched_at": p.fetched_at,
            }
            for p in pages
        ]

    records = asyncio.run(run(), debug=True)
    assert len(records) == 3
    assert "/secret" not in fetched_paths
    assert fetched_paths.count("/p1") == 1

    jsonl_path = tmp_path / "crawl.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )

    # Verify Lab 01/02/05 indexer consumes the crawled JSONL stream seamlessly
    idx = build_index(jsonl_path, executor="serial")
    engine = SearchEngine(idx)
    hits = engine.search("coroutines OR event OR search", k=5)
    assert len(hits) >= 1


def test_cli_crawl_help() -> None:
    res = runner.invoke(app, ["crawl", "--help"])
    assert res.exit_code == 0
    assert "--max-pages" in res.stdout
    assert "--concurrency" in res.stdout
    assert "--per-host" in res.stdout
