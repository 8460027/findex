from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "findex-crawler/0.6 (+https://github.com/8460027/findex)"


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    final_url: str
    status: int
    html: str
    bytes_len: int
    elapsed: float
    attempts: int
    error: str | None = None


def _parse_retry_after(header_value: str | None, *, max_wait: float = 30.0) -> float | None:
    if not header_value:
        return None
    raw = header_value.strip()
    try:
        seconds = float(raw)
        return max(0.0, min(seconds, max_wait))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        delta = (dt - datetime.now(UTC)).total_seconds()
        return max(0.0, min(delta, max_wait))
    except (TypeError, ValueError):
        return None


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    base_backoff: float = 0.25,
    max_backoff: float = 8.0,
) -> FetchResult:
    """Politely fetch a URL with asyncio.timeout, exponential backoff + jitter, and Retry-After."""
    t0 = time.perf_counter()
    last_status = 0
    last_error: str | None = None
    last_final_url = url

    for attempt in range(max_retries + 1):
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url, follow_redirects=True)
                body = resp.text
                raw_len = len(resp.content)
                last_status = resp.status_code
                last_final_url = str(resp.url)

            if 200 <= resp.status_code < 300:
                elapsed = time.perf_counter() - t0
                log.info("GET %s -> %d (%d B, %.3fs)", url, resp.status_code, raw_len, elapsed)
                return FetchResult(
                    url=url,
                    final_url=last_final_url,
                    status=resp.status_code,
                    html=body,
                    bytes_len=raw_len,
                    elapsed=elapsed,
                    attempts=attempt + 1,
                )

            # Never retry 4xx except 429 Too Many Requests
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                elapsed = time.perf_counter() - t0
                log.warning("GET %s -> %d (no retry, %.3fs)", url, resp.status_code, elapsed)
                return FetchResult(
                    url=url,
                    final_url=last_final_url,
                    status=resp.status_code,
                    html="",
                    bytes_len=raw_len,
                    elapsed=elapsed,
                    attempts=attempt + 1,
                    error=f"HTTP {resp.status_code}",
                )

            last_error = f"HTTP {resp.status_code}"
            if attempt >= max_retries:
                break

            retry_after = (
                _parse_retry_after(resp.headers.get("Retry-After"))
                if resp.status_code == 429
                else None
            )
            delay = (
                retry_after
                if retry_after is not None
                else min(max_backoff, base_backoff * (2**attempt)) + random.uniform(0.0, 0.25)
            )
            log.warning(
                "GET %s -> %d (attempt %d/%d, retrying in %.2fs)",
                url,
                resp.status_code,
                attempt + 1,
                max_retries + 1,
                delay,
            )
            await asyncio.sleep(delay)

        except (TimeoutError, httpx.TransportError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= max_retries:
                break
            delay = min(max_backoff, base_backoff * (2**attempt)) + random.uniform(0.0, 0.25)
            log.warning(
                "GET %s -> %s (attempt %d/%d, retrying in %.2fs)",
                url,
                type(exc).__name__,
                attempt + 1,
                max_retries + 1,
                delay,
            )
            await asyncio.sleep(delay)

    elapsed = time.perf_counter() - t0
    log.error("GET %s -> FAILED after %d attempts (%s)", url, max_retries + 1, last_error)
    return FetchResult(
        url=url,
        final_url=last_final_url,
        status=last_status,
        html="",
        bytes_len=0,
        elapsed=elapsed,
        attempts=max_retries + 1,
        error=last_error,
    )
