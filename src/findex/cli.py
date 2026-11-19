from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from findex.crawler.crawl import CrawlStats, crawl
from findex.index import BuildTiming, ExecutorType, build_index
from findex.query import QuerySyntaxError
from findex.search import SearchEngine
from findex.stats import run_lazy_stats
from findex.store import FormatType, open_index, save_index

log = logging.getLogger(__name__)

stdout_console = Console()
stderr_console = Console(stderr=True)

app = typer.Typer(
    name="findex",
    help="Fast streaming search engine with BM25/TF-IDF ranking and Boolean/phrase queries.",
    no_args_is_help=True,
    add_completion=True,
)


class ScorerChoice(StrEnum):
    bm25 = "bm25"
    tfidf = "tfidf"


class FormatChoice(StrEnum):
    binary = "binary"
    json = "json"
    pickle = "pickle"


class ExecutorChoice(StrEnum):
    serial = "serial"
    threads = "threads"
    processes = "processes"


def _configure_logging(verbosity: int) -> None:
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(levelname)s [%(name)s]: %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _format_snippet_rich(snippet: str) -> str:
    """Escape raw document text and highlight [matched] tokens in bold yellow."""
    parts: list[str] = []
    last = 0
    for match in re.finditer(r"\[([^\[\]]+)\]", snippet):
        parts.append(escape(snippet[last : match.start()]))
        parts.append(f"[bold yellow]{escape(match.group(1))}[/bold yellow]")
        last = match.end()
    parts.append(escape(snippet[last:]))
    return "".join(parts)


@app.callback()
def main_callback(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase diagnostic verbosity (-v for INFO, -vv for DEBUG).",
        ),
    ] = 0,
) -> None:
    """Configure global CLI logging on stderr."""
    _configure_logging(verbose)


@app.command("index")
def index_cmd(
    corpus: Annotated[Path, typer.Argument(help="Path to corpus directory or .jsonl file.")],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Output path for the serialized index.")
    ] = Path("index.bin"),
    fmt: Annotated[
        FormatChoice, typer.Option("--format", "-f", help="Index serialization format.")
    ] = FormatChoice.binary,
    positions: Annotated[
        bool, typer.Option("--positions", help="Store token positions for phrase queries.")
    ] = False,
    limit: Annotated[
        int | None, typer.Option("--limit", "-l", help="Index at most N documents.")
    ] = None,
    executor: Annotated[
        ExecutorChoice,
        typer.Option("--executor", "-e", help="Execution backend (serial, threads, processes)."),
    ] = ExecutorChoice.processes,
    workers: Annotated[
        int | None,
        typer.Option("--workers", "-w", min=1, help="Number of worker threads/processes."),
    ] = None,
) -> None:
    """Build and save an inverted index from a document corpus."""
    if not corpus.exists():
        stderr_console.print(f"[bold red]Error:[/bold red] Corpus path not found: {corpus}")
        raise typer.Exit(code=1)

    t0 = time.perf_counter()
    timings: BuildTiming = {"map_seconds": 0.0, "merge_seconds": 0.0, "total_seconds": 0.0}
    try:
        exec_backend: ExecutorType = executor.value  # type: ignore[assignment]  # StrEnum matches ExecutorType literals
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=stderr_console,
            transient=True,
        ) as progress:
            progress.add_task(f"Indexing {corpus} ({exec_backend})...", total=None)
            idx = build_index(
                corpus,
                with_positions=positions,
                limit=limit,
                executor=exec_backend,
                workers=workers,
                timings_out=timings,
            )
            save_index(idx, out, fmt=fmt.value)  # type: ignore[arg-type]  # StrEnum matches FormatType literals
    except OSError as exc:
        stderr_console.print(f"[bold red]Error:[/bold red] Failed to build/save index: {exc}")
        raise typer.Exit(code=1) from exc

    elapsed = time.perf_counter() - t0
    size_mb = out.stat().st_size / (1024 * 1024)
    log.info(
        "Indexed %d docs (%d terms) into %s (%.2f MB) in %.3f s [map=%.3fs, merge=%.3fs]",
        idx.num_docs,
        len(idx),
        out,
        size_mb,
        elapsed,
        timings["map_seconds"],
        timings["merge_seconds"],
    )
    stdout_console.print(
        f"[green]Indexed[/green] [bold]{idx.num_docs}[/bold] documents "
        f"([bold]{len(idx)}[/bold] unique terms) -> [cyan]{out}[/cyan] "
        f"({size_mb:.2f} MB in {elapsed:.3f} s, executor={executor.value})"
    )


@app.command("search")
def search_cmd(
    index_path: Annotated[Path, typer.Argument(help="Path to saved index file.")],
    query: Annotated[str, typer.Argument(help="Boolean, phrase, or ranked search query.")],
    k: Annotated[int, typer.Option("--k", "-k", min=1, help="Number of top results.")] = 10,
    scorer: Annotated[
        ScorerChoice, typer.Option("--scorer", "-s", help="Ranking function (bm25 or tfidf).")
    ] = ScorerChoice.bm25,
    fmt: Annotated[
        FormatChoice, typer.Option("--format", "-f", help="Index serialization format.")
    ] = FormatChoice.binary,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit results as JSON lines to stdout.")
    ] = False,
) -> None:
    """Search a saved index and display ranked results with highlighted snippets."""
    if not index_path.exists():
        stderr_console.print(f"[bold red]Error:[/bold red] Index file not found: {index_path}")
        raise typer.Exit(code=1)

    t0 = time.perf_counter()
    try:
        index_fmt: FormatType = fmt.value  # type: ignore[assignment]  # StrEnum matches FormatType literals
        with open_index(index_path, fmt=index_fmt) as idx:
            engine = SearchEngine(idx)
            results = engine.search(query, scorer=scorer.value, k=k, with_snippets=True)
    except (QuerySyntaxError, ValueError, OSError) as exc:
        stderr_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    log.info(
        "Query %r (%s) returned %d hits in %.2f ms",
        query,
        scorer.value,
        len(results),
        elapsed_ms,
    )

    if json_output:
        for rank, res in enumerate(results, start=1):
            record = {
                "rank": rank,
                "doc_id": res.doc_id,
                "title": res.title,
                "score": round(res.score, 6),
                "snippet": res.snippet,
            }
            sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    if not results:
        stdout_console.print(f"[yellow]No documents matched query:[/yellow] {escape(query)}")
        return

    table = Table(
        title=f"Search Results for {query!r} (scorer={scorer.value}, {elapsed_ms:.2f} ms)",
        show_lines=True,
    )
    table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
    table.add_column("Doc ID", justify="right", style="magenta", no_wrap=True)
    table.add_column("Title", style="bold green")
    table.add_column("Score", justify="right", style="yellow", no_wrap=True)
    table.add_column("Snippet")

    for rank, res in enumerate(results, start=1):
        table.add_row(
            str(rank),
            str(res.doc_id),
            escape(res.title),
            f"{res.score:.4f}",
            _format_snippet_rich(res.snippet),
        )

    stdout_console.print(table)


@app.command("stats")
def stats_cmd(
    target: Annotated[
        Path, typer.Argument(help="Path to corpus directory/.jsonl or saved index file.")
    ],
    limit: Annotated[
        int | None, typer.Option("--limit", "-l", help="Limit documents when scanning a corpus.")
    ] = None,
    fmt: Annotated[
        FormatChoice, typer.Option("--format", "-f", help="Index format (if target is an index).")
    ] = FormatChoice.binary,
    top_n: Annotated[int, typer.Option("--top", help="Number of top terms to show.")] = 10,
) -> None:
    """Display summary statistics for a corpus directory or a built index file."""
    if not target.exists():
        stderr_console.print(f"[bold red]Error:[/bold red] Target path not found: {target}")
        raise typer.Exit(code=1)

    if target.is_dir() or target.suffix == ".jsonl":
        stats = run_lazy_stats(target, limit=limit)
        log.info(
            "Computed corpus stats in %.3f s (peak %.2f MB)", stats["elapsed"], stats["peak_mb"]
        )

        summary = Table(title=f"Corpus Statistics: {target}")
        summary.add_column("Metric", style="bold cyan")
        summary.add_column("Value", justify="right", style="green")
        summary.add_row("Documents", f"{stats['doc_count']:,}")
        summary.add_row("Total tokens", f"{stats['total_tokens']:,}")
        summary.add_row("Vocabulary size", f"{stats['vocab_size']:,}")
        summary.add_row("Elapsed time", f"{stats['elapsed']:.3f} s")
        summary.add_row("Peak memory", f"{stats['peak_mb']:.2f} MB")
        stdout_console.print(summary)

        top_table = Table(title=f"Top {min(top_n, len(stats['top_50']))} Terms (by TF)")
        top_table.add_column("Rank", justify="right", style="cyan")
        top_table.add_column("Term", style="bold")
        top_table.add_column("Count", justify="right", style="yellow")
        for rank, (term, count) in enumerate(stats["top_50"][:top_n], start=1):
            top_table.add_row(str(rank), escape(term), f"{count:,}")
        stdout_console.print(top_table)
        return

    try:
        index_fmt: FormatType = fmt.value  # type: ignore[assignment]  # StrEnum matches FormatType literals
        with open_index(target, fmt=index_fmt) as idx:
            summary = Table(title=f"Index Statistics: {target}")
            summary.add_column("Metric", style="bold cyan")
            summary.add_column("Value", justify="right", style="green")
            summary.add_row("Indexed documents", f"{idx.num_docs:,}")
            summary.add_row("Unique terms", f"{len(idx):,}")
            summary.add_row("Average doc length", f"{idx.avg_doc_length:.2f} tokens")
            stdout_console.print(summary)

            ranked_df = sorted(
                ((term, len(plist)) for term, plist in idx.postings.items()),
                key=lambda item: item[1],
                reverse=True,
            )[:top_n]
            top_table = Table(title=f"Top {len(ranked_df)} Terms (by Document Frequency)")
            top_table.add_column("Rank", justify="right", style="cyan")
            top_table.add_column("Term", style="bold")
            top_table.add_column("DF", justify="right", style="yellow")
            for rank, (term, df_val) in enumerate(ranked_df, start=1):
                top_table.add_row(str(rank), escape(term), f"{df_val:,}")
            stdout_console.print(top_table)
    except (ValueError, TypeError, OSError) as exc:
        stderr_console.print(f"[bold red]Error:[/bold red] Failed to inspect index: {exc}")
        raise typer.Exit(code=1) from exc


@app.command("crawl")
def crawl_cmd(
    seed: Annotated[str, typer.Argument(help="Seed URL to start crawling from.")],
    max_pages: Annotated[
        int, typer.Option("--max-pages", "-n", min=1, help="Maximum number of pages to fetch.")
    ] = 500,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-c", min=1, help="Maximum concurrent requests.")
    ] = 10,
    per_host: Annotated[
        int, typer.Option("--per-host", min=1, help="Maximum concurrent requests per host.")
    ] = 2,
    delay: Annotated[
        float,
        typer.Option("--delay", min=0.0, help="Minimum delay (s) between requests to same host."),
    ] = 0.1,
    domain: Annotated[
        list[str] | None,
        typer.Option("--domain", "-d", help="Allowed domain(s) (defaults to seed host)."),
    ] = None,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output JSONL corpus file path.")] = Path(
        "data/crawl.jsonl"
    ),
    log_file: Annotated[Path, typer.Option("--log", help="Per-URL crawl log file path.")] = Path(
        "crawl.log"
    ),
) -> None:
    """Politely crawl web pages asynchronously and stream them to a JSONL corpus."""
    out.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    crawler_logger = logging.getLogger("findex.crawler")
    crawler_logger.setLevel(logging.INFO)
    crawler_logger.addHandler(file_handler)

    stats = CrawlStats()
    t0 = time.perf_counter()

    def _render_panel() -> Panel:
        elapsed = max(0.001, time.perf_counter() - t0)
        rate = stats.pages_fetched / elapsed
        mb = stats.bytes_downloaded / (1024 * 1024)
        body = (
            f"[bold cyan]Seed:[/bold cyan] {escape(seed)}\n"
            f"[bold green]Pages fetched:[/bold green] {stats.pages_fetched} / {max_pages}  "
            f"([bold]{rate:.1f} pages/s[/bold], {mb:.2f} MB)\n"
            f"[bold yellow]In-flight:[/bold yellow] {stats.in_flight}  |  "
            f"[bold magenta]Queue depth:[/bold magenta] {stats.queue_depth}  |  "
            f"[bold red]Errors:[/bold red] {stats.errors}"
        )
        return Panel(body, title="findex async crawler", border_style="cyan")

    async def _run_crawl() -> None:
        with (
            out.open("w", encoding="utf-8") as f,
            Live(_render_panel(), console=stderr_console, refresh_per_second=10) as live,
        ):
            async for page in crawl(
                [seed],
                max_pages=max_pages,
                concurrency=concurrency,
                per_host=per_host,
                per_host_delay=delay,
                allowed_domains=domain,
                stats=stats,
            ):
                record = {
                    "id": page.url,
                    "url": page.url,
                    "title": page.title,
                    "text": page.text,
                    "fetched_at": page.fetched_at,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                live.update(_render_panel())

    try:
        asyncio.run(_run_crawl())
    finally:
        crawler_logger.removeHandler(file_handler)
        file_handler.close()

    elapsed = time.perf_counter() - t0
    rate = stats.pages_fetched / max(0.001, elapsed)
    stdout_console.print(
        f"[green]Crawled[/green] [bold]{stats.pages_fetched}[/bold] pages in {elapsed:.2f} s "
        f"([bold]{rate:.1f} pages/s[/bold], errors={stats.errors}) -> [cyan]{out}[/cyan]"
    )


if __name__ == "__main__":
    app()
