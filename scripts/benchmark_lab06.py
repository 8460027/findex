from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


async def _serve_mirror(
    total_pages: int = 260, latency_s: float = 0.025
) -> tuple[asyncio.Server, int]:
    rate_limited_count = 0

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw_line = await reader.readline()
            if not raw_line:
                return
            parts = raw_line.decode("latin1").strip().split()
            path = parts[1] if len(parts) >= 2 else "/"
            while True:
                hdr = await reader.readline()
                if not hdr or hdr == b"\r\n":
                    break

            await asyncio.sleep(latency_s)

            if path == "/robots.txt":
                body = b"User-agent: *\nAllow: /\nDisallow: /admin\n"
                status_line = b"HTTP/1.1 200 OK\r\n"
                extra_headers = b"Content-Type: text/plain; charset=utf-8\r\n"
            nonlocal rate_limited_count
            if path == "/robots.txt":
                body = b"User-agent: *\nAllow: /\nDisallow: /admin\n"
                status_line = b"HTTP/1.1 200 OK\r\n"
                extra_headers = b"Content-Type: text/plain; charset=utf-8\r\n"
            elif path == "/page/7" and rate_limited_count < 3:
                rate_limited_count += 1
                body = b"Too Many Requests"
                status_line = b"HTTP/1.1 429 Too Many Requests\r\n"
                extra_headers = b"Retry-After: 0.05\r\nContent-Type: text/plain\r\n"
            elif path == "/" or path.startswith("/page/"):
                idx = 0 if path == "/" else int(path.removeprefix("/page/"))
                if 0 <= idx < total_pages:
                    links = " ".join(
                        f"<a href='/page/{nxt}'>Doc {nxt}</a>"
                        for nxt in (
                            (idx * 3 + 1) % total_pages,
                            (idx * 3 + 2) % total_pages,
                            (idx + 1) % total_pages,
                        )
                    )
                    html = (
                        f"<html><head><title>Mirror Document {idx}</title></head><body>"
                        f"<h1>Document {idx}</h1>"
                        f"<p>Asyncio cooperative multitasking event loop coroutine worker page {idx}.</p>"
                        f"{links}</body></html>"
                    )
                    body = html.encode("utf-8")
                    status_line = b"HTTP/1.1 200 OK\r\n"
                    extra_headers = b"Content-Type: text/html; charset=utf-8\r\n"
                else:
                    body = b"Not Found"
                    status_line = b"HTTP/1.1 404 Not Found\r\n"
                    extra_headers = b"Content-Type: text/plain\r\n"
            else:
                body = b"Not Found"
                status_line = b"HTTP/1.1 404 Not Found\r\n"
                extra_headers = b"Content-Type: text/plain\r\n"

            response = (
                status_line
                + extra_headers
                + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("latin1")
                + body
            )
            writer.write(response)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", 0, backlog=256)
    sock = server.sockets[0]
    port = int(sock.getsockname()[1])
    return server, port


def _run_subprocess_bench(
    port: int, concurrency: int, max_pages: int, out_jsonl: Path, log_path: Path
) -> dict[str, object]:
    code = f"""
import asyncio, json, logging, time, tracemalloc
from pathlib import Path
from findex.crawler.crawl import CrawlStats, crawl

fh = logging.FileHandler({str(log_path)!r}, mode="w", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
lg = logging.getLogger("findex.crawler")
lg.setLevel(logging.INFO)
lg.addHandler(fh)

async def main():
    tracemalloc.start()
    stats = CrawlStats()
    t0 = time.perf_counter()
    with Path({str(out_jsonl)!r}).open("w", encoding="utf-8") as f:
        async for p in crawl(
            ["http://127.0.0.1:{port}/"],
            max_pages={max_pages},
            concurrency={concurrency},
            per_host={concurrency},
            per_host_delay=0.0,
            stats=stats,
        ):
            f.write(json.dumps({{"id": p.url, "url": p.url, "title": p.title, "text": p.text}}) + "\\n")
    wall = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory(); peak_mb = peak_bytes / (1024 * 1024)
    print(json.dumps({{"pages": stats.pages_fetched, "wall": wall, "errors": stats.errors, "peak_mb": peak_mb}}))

asyncio.run(main())
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_SRC)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


async def main() -> None:
    server, port = await _serve_mirror(total_pages=260, latency_s=0.025)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            print(
                "=== Lab 06 Async Crawler Benchmark (220 pages, ~25 ms/req simulated network latency) ==="
            )
            for conc in (1, 5, 20):
                out_jsonl = tmp_dir / f"crawl_c{conc}.jsonl"
                log_path = tmp_dir / f"crawl_c{conc}.log"
                res = await asyncio.to_thread(
                    _run_subprocess_bench, port, conc, 220, out_jsonl, log_path
                )
                pages = int(res["pages"])
                wall = float(res["wall"])
                errors = int(res["errors"])
                peak_mb = float(res["peak_mb"])
                rate = pages / max(0.001, wall)
                print(
                    f"concurrency={conc:2d} | pages={pages:3d} | wall={wall:6.3f}s | "
                    f"pages/s={rate:6.1f} | errors={errors} | peak_rss={peak_mb:5.1f}MB"
                )
                if conc == 20:
                    print("\n=== Sample Handled 429 + Retry-After Log Lines ===")
                    for line in log_path.read_text(encoding="utf-8").splitlines():
                        if "/page/7" in line or "robots.txt" in line:
                            print(line)
    finally:
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
