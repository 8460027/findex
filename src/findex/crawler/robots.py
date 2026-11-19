from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import USER_AGENT

log = logging.getLogger(__name__)


class RobotsCache:
    """Fetch, cache, and evaluate robots.txt rules per origin (scheme + netloc)."""

    def __init__(self, client: httpx.AsyncClient, *, user_agent: str = USER_AGENT) -> None:
        self._client = client
        self._user_agent = user_agent
        self._parsers: dict[str, RobotFileParser | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False

        origin = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
        lock = self._locks.setdefault(origin, asyncio.Lock())

        async with lock:
            if origin not in self._parsers:
                self._parsers[origin] = await self._load_robots(origin)

        parser = self._parsers[origin]
        if parser is None:
            return True
        return parser.can_fetch(self._user_agent, url)

    async def _load_robots(self, origin: str) -> RobotFileParser | None:
        robots_url = f"{origin}/robots.txt"
        try:
            async with asyncio.timeout(5.0):
                resp = await self._client.get(robots_url, follow_redirects=True)
            if resp.status_code == 200:
                rp = RobotFileParser()
                rp.set_url(robots_url)
                rp.parse(resp.text.splitlines())
                log.info("Loaded robots.txt for %s", origin)
                return rp
            if 400 <= resp.status_code < 500:
                log.info("No robots.txt at %s (%d); allowing crawl", robots_url, resp.status_code)
                return None
            log.warning(
                "robots.txt at %s returned %d; allowing crawl", robots_url, resp.status_code
            )
            return None
        except (TimeoutError, httpx.TransportError) as exc:
            log.warning("Failed to fetch %s (%s); allowing crawl", robots_url, type(exc).__name__)
            return None
