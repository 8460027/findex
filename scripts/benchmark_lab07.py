from __future__ import annotations

import asyncio
import json
import math
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

import httpx

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, math.ceil((pct / 100.0) * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def _build_synthetic_corpus(path: Path, num_docs: int = 800) -> None:
    topics = [
        "python asyncio event loop coroutine taskgroup semaphore await",
        "fastapi pydantic validation asgi uvicorn lifespan dependency injection",
        "bm25 tfidf inverted index postings ranking query parser boolean phrase",
        "multiprocessing gil threadpool processpool map reduce concurrency",
    ]
    lines: list[str] = []
    for i in range(num_docs):
        body = (topics[i % len(topics)] + f" document_{i} ") * 25
        lines.append(json.dumps({"id": f"doc-{i}", "title": f"Technical Note {i}", "text": body}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def _wait_healthy(base_url: str, timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(f"{base_url}/health", timeout=1.0)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                await asyncio.sleep(0.1)
                continue
            await asyncio.sleep(0.1)
    msg = f"Server at {base_url} did not become healthy within {timeout_s}s"
    raise TimeoutError(msg)


async def _run_load(
    base_urls: str | list[str],
    *,
    concurrency: int = 50,
    total_requests: int = 300,
    endpoint: str = "/search",
) -> dict[str, float]:
    queries = [
        "python AND asyncio",
        "bm25 OR tfidf",
        "fastapi pydantic",
        '"event loop"',
        "gil concurrency",
    ]
    latencies_ms: list[float] = []
    sem = asyncio.Semaphore(concurrency)

    urls = [base_urls] if isinstance(base_urls, str) else base_urls
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency),
    ) as client:
        for u in urls:
            for q in queries:
                await client.get(f"{u}{endpoint}", params={"q": q, "k": 10})

        t_start = time.perf_counter()

        async def one_req(i: int) -> None:
            q = queries[i % len(queries)]
            async with sem:
                t0 = time.perf_counter()
                target = urls[i % len(urls)]
                resp = await client.get(
                    f"{target}{endpoint}", params={"q": q, "k": 10}, timeout=30.0
                )
                resp.raise_for_status()
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        async with asyncio.TaskGroup() as tg:
            for i in range(total_requests):
                tg.create_task(one_req(i))

        wall_s = time.perf_counter() - t_start

    latencies_ms.sort()
    return {
        "rps": total_requests / max(0.001, wall_s),
        "p50": _percentile(latencies_ms, 50),
        "p95": _percentile(latencies_ms, 95),
        "p99": _percentile(latencies_ms, 99),
        "wall_s": wall_s,
    }


async def _stop_proc(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=30.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        corpus_path = tmp_dir / "bench_corpus.jsonl"
        _build_synthetic_corpus(corpus_path, num_docs=800)

        wrapper_mod = tmp_dir / "bench_asgi_app.py"
        wrapper_mod.write_text(
            """
import math
import time
from typing import Annotated, Literal

from fastapi import Depends, Query

from findex.search import SearchEngine
from findex.web.app import app, build_snippet, get_engine, parse_query
from findex.web.schemas import SearchResponse, SearchResultItem


@app.get("/search-async", response_model=SearchResponse)
async def search_async(
    q: Annotated[str, Query(min_length=1, max_length=200)],
    k: Annotated[int, Query(ge=1, le=100)] = 10,
    scorer: Annotated[Literal["bm25", "tfidf"], Query()] = "bm25",
    page: Annotated[int, Query(ge=1, le=1000)] = 1,
    engine: Annotated[SearchEngine, Depends(get_engine)] = ...,  # type: ignore[assignment]
) -> SearchResponse:
    t0 = time.perf_counter()
    ast = parse_query(q)
    matched = ast.evaluate(engine.index)
    total = len(matched)
    all_top = engine.search(q, scorer=scorer, k=max(page * k, k), with_snippets=False)
    slice_hits = all_top[(page - 1) * k : page * k]
    terms = ast.positive_terms()
    items = [
        SearchResultItem(
            doc_id=h.doc_id,
            title=h.title,
            score=round(h.score, 6),
            snippet=build_snippet(engine.index.doc_meta[h.doc_id].path, terms),
        )
        for h in slice_hits
    ]
    return SearchResponse(
        query=q,
        scorer=scorer,
        page=page,
        k=k,
        total=total,
        total_pages=math.ceil(total / k) if total else 0,
        took_ms=round((time.perf_counter() - t0) * 1000.0, 3),
        results=items,
    )
""",
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{_SRC}{os.pathsep}{tmp_dir}"
        env["INDEX_PATH"] = str(corpus_path)
        env["LOG_LEVEL"] = "ERROR"

        port1 = _free_port()
        proc1 = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "uvicorn",
            "bench_asgi_app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port1),
            "--log-level",
            "error",
            env=env,
        )
        try:
            await _wait_healthy(f"http://127.0.0.1:{port1}")
            c20_async = await _run_load(
                f"http://127.0.0.1:{port1}",
                concurrency=20,
                total_requests=200,
                endpoint="/search-async",
            )
            c20_def = await _run_load(
                f"http://127.0.0.1:{port1}",
                concurrency=20,
                total_requests=200,
                endpoint="/search",
            )
            c50_async = await _run_load(
                f"http://127.0.0.1:{port1}",
                concurrency=50,
                total_requests=300,
                endpoint="/search-async",
            )
            c50_def = await _run_load(
                f"http://127.0.0.1:{port1}",
                concurrency=50,
                total_requests=300,
                endpoint="/search",
            )
        finally:
            await _stop_proc(proc1)

        w4_ports = [_free_port() for _ in range(4)]
        w4_procs: list[asyncio.subprocess.Process] = []
        try:
            for p in w4_ports:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "findex.web:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(p),
                    "--log-level",
                    "error",
                    env=env,
                )
                w4_procs.append(proc)
            w4_urls = [f"http://127.0.0.1:{p}" for p in w4_ports]
            for u in w4_urls:
                await _wait_healthy(u)
            c50_w4 = await _run_load(
                w4_urls,
                concurrency=50,
                total_requests=300,
                endpoint="/search",
            )
        finally:
            for proc in w4_procs:
                await _stop_proc(proc)

        print("=== M2: 20 Concurrent Requests (async def vs def on 1 worker) ===")
        print(
            f"async def (c=20): wall={c20_async['wall_s']:.3f}s | RPS={c20_async['rps']:.1f} | "
            f"p50={c20_async['p50']:.2f}ms | p99={c20_async['p99']:.2f}ms"
        )
        print(
            f"def       (c=20): wall={c20_def['wall_s']:.3f}s | RPS={c20_def['rps']:.1f} | "
            f"p50={c20_def['p50']:.2f}ms | p99={c20_def['p99']:.2f}ms"
        )
        print("\n=== M4: Load Test Table (c=50, 600 requests) ===")
        print(
            f"local, 1 worker, async def | c=50 | RPS={c50_async['rps']:.1f} | "
            f"p50={c50_async['p50']:.2f}ms | p95={c50_async['p95']:.2f}ms | p99={c50_async['p99']:.2f}ms"
        )
        print(
            f"local, 1 worker, def       | c=50 | RPS={c50_def['rps']:.1f} | "
            f"p50={c50_def['p50']:.2f}ms | p95={c50_def['p95']:.2f}ms | p99={c50_def['p99']:.2f}ms"
        )
        print(
            f"local, 4 workers, def      | c=50 | RPS={c50_w4['rps']:.1f} | "
            f"p50={c50_w4['p50']:.2f}ms | p95={c50_w4['p95']:.2f}ms | p99={c50_w4['p99']:.2f}ms"
        )


if __name__ == "__main__":
    asyncio.run(main())
