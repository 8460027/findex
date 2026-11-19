from __future__ import annotations

import asyncio
import importlib
import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser as StdHTMLParser
from typing import Any
from urllib.parse import parse_qsl, urldefrag, urlencode, urljoin, urlparse, urlunparse

import httpx

from findex.crawler.fetch import USER_AGENT, fetch
from findex.crawler.robots import RobotsCache

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str


@dataclass(slots=True)
class CrawlStats:
    pages_fetched: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_depth: int = 0
    bytes_downloaded: int = 0


def canonicalize_url(raw_url: str, *, base_url: str | None = None) -> str | None:
    """Normalize a URL (lowercase scheme/host, strip fragment, sort query, clean trailing slash)."""
    joined = urljoin(base_url, raw_url.strip()) if base_url else raw_url.strip()
    clean, _ = urldefrag(joined)
    parsed = urlparse(clean)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    if scheme not in {"http", "https"} or not netloc:
        return None

    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    sorted_query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    return urlunparse((scheme, netloc, path, "", sorted_query, ""))


class _FallbackHTMLParser(StdHTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.skip_depth = 0
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1
        elif t == "title":
            self.in_title = True
        elif t == "a":
            for k, v in attrs:
                if k.lower() == "href" and v:
                    self.hrefs.append(v)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"script", "style", "noscript", "svg"} and self.skip_depth > 0:
            self.skip_depth -= 1
        elif t == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.skip_depth > 0:
            return
        if self.in_title:
            self.title_parts.append(data)
        self.text_parts.append(data)


def extract_links_and_text(html: str, base_url: str) -> tuple[str, str, list[str]]:
    """Extract page title, clean visible text, and canonical outgoing links."""
    try:
        slx: Any = importlib.import_module("selectolax.parser")
        tree: Any = slx.HTMLParser(html)
        title_node: Any = tree.css_first("title")
        title = str(title_node.text(strip=True)) if title_node is not None else base_url

        links: list[str] = []
        for node in tree.css("a[href]"):
            href: object = node.attributes.get("href")
            if isinstance(href, str) and href:
                canon = canonicalize_url(href, base_url=base_url)
                if canon is not None:
                    links.append(canon)

        tree.strip_tags(["script", "style", "noscript", "svg"])
        body: Any = tree.body
        raw_text = (
            str(body.text(separator=" ", strip=True))
            if body is not None
            else str(tree.text(separator=" ", strip=True))
        )
        return title, " ".join(raw_text.split()), links
    except (ImportError, AttributeError, RuntimeError):
        parser = _FallbackHTMLParser()
        parser.feed(html)
        title = " ".join("".join(parser.title_parts).split()) or base_url
        links = []
        for href in parser.hrefs:
            canon = canonicalize_url(href, base_url=base_url)
            if canon is not None:
                links.append(canon)
        return title, " ".join(" ".join(parser.text_parts).split()), links


async def crawl(
    seeds: Sequence[str],
    *,
    max_pages: int = 500,
    concurrency: int = 10,
    per_host: int = 2,
    per_host_delay: float = 0.1,
    allowed_domains: Sequence[str] | None = None,
    timeout: float = 10.0,
    client: httpx.AsyncClient | None = None,
    stats: CrawlStats | None = None,
) -> AsyncIterator[Page]:
    """Async generator yielding crawled Pages concurrently using Queue + TaskGroup."""
    if max_pages <= 0 or not seeds:
        return

    crawl_stats = stats if stats is not None else CrawlStats()
    frontier: asyncio.Queue[str] = asyncio.Queue()
    out_queue: asyncio.Queue[Page | None] = asyncio.Queue()
    seen: set[str] = set()

    normalized_seeds: list[str] = []
    for seed in seeds:
        canon = canonicalize_url(seed)
        if canon is not None and canon not in seen:
            seen.add(canon)
            normalized_seeds.append(canon)

    if not normalized_seeds:
        return

    domain_allowlist: set[str] = (
        {d.lower() for d in allowed_domains}
        if allowed_domains
        else {urlparse(u).netloc.lower() for u in normalized_seeds}
    )

    def _is_domain_allowed(url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return any(host == d or host.endswith(f".{d}") for d in domain_allowlist)

    for u in normalized_seeds:
        if _is_domain_allowed(u):
            frontier.put_nowait(u)

    crawl_stats.queue_depth = frontier.qsize()
    if frontier.empty():
        return

    global_sem = asyncio.Semaphore(max(1, concurrency))
    host_sems: dict[str, asyncio.Semaphore] = {}
    host_rate_locks: dict[str, asyncio.Lock] = {}
    host_last_request: dict[str, float] = {}
    stop_event = asyncio.Event()
    claim_lock = asyncio.Lock()
    claimed_pages = 0
    active_workers = 0

    owns_client = client is None
    http_client = (
        httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            limits=httpx.Limits(
                max_connections=max(10, concurrency * 2),
                max_keepalive_connections=concurrency,
            ),
        )
        if owns_client
        else client
    )
    robots = RobotsCache(http_client)

    async def _rate_limit_host(host: str) -> None:
        if per_host_delay <= 0:
            return
        lock = host_rate_locks.setdefault(host, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            last = host_last_request.get(host)
            if last is not None:
                wait = per_host_delay - (now - last)
                if wait > 0:
                    await asyncio.sleep(wait)
            host_last_request[host] = time.monotonic()

    async def _worker() -> None:
        nonlocal claimed_pages, active_workers
        while not stop_event.is_set():
            try:
                async with asyncio.timeout(0.05):
                    url = await frontier.get()
            except TimeoutError:
                if frontier.empty() and active_workers == 0:
                    stop_event.set()
                    return
                continue

            active_workers += 1
            crawl_stats.queue_depth = frontier.qsize()
            try:
                if stop_event.is_set() or not _is_domain_allowed(url):
                    continue

                if not await robots.allowed(url):
                    log.info("Blocked by robots.txt: %s", url)
                    continue

                async with claim_lock:
                    if claimed_pages >= max_pages:
                        stop_event.set()
                        continue
                    claimed_pages += 1

                host = urlparse(url).netloc.lower()
                host_sem = host_sems.setdefault(host, asyncio.Semaphore(max(1, per_host)))

                async with global_sem, host_sem:
                    crawl_stats.in_flight += 1
                    try:
                        await _rate_limit_host(host)
                        res = await fetch(http_client, url, timeout=timeout)
                    finally:
                        crawl_stats.in_flight -= 1

                if res.error is not None or not res.html:
                    crawl_stats.errors += 1
                    async with claim_lock:
                        claimed_pages -= 1
                    continue

                crawl_stats.bytes_downloaded += res.bytes_len
                title, text, links = extract_links_and_text(res.html, res.final_url)

                if not stop_event.is_set():
                    for link in links:
                        if link not in seen and _is_domain_allowed(link):
                            seen.add(link)
                            frontier.put_nowait(link)

                page = Page(
                    url=res.final_url,
                    title=title,
                    text=text,
                    fetched_at=datetime.now(UTC).isoformat(),
                )
                crawl_stats.pages_fetched += 1
                await out_queue.put(page)

                if crawl_stats.pages_fetched >= max_pages:
                    stop_event.set()
            finally:
                active_workers -= 1
                frontier.task_done()
                crawl_stats.queue_depth = frontier.qsize()

    async def _supervisor() -> None:
        try:
            async with asyncio.TaskGroup() as tg:
                for _ in range(max(1, concurrency)):
                    tg.create_task(_worker())
        finally:
            await out_queue.put(None)

    sup_task = asyncio.create_task(_supervisor())
    try:
        while True:
            item = await out_queue.get()
            if item is None:
                break
            yield item
    finally:
        stop_event.set()
        if not sup_task.done():
            sup_task.cancel()
        await asyncio.gather(sup_task, return_exceptions=True)
        if owns_client:
            await http_client.aclose()
