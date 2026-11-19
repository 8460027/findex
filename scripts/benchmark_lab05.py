from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from findex.index import BuildTiming, ExecutorType, build_index, collect_doc_refs
from findex.tokenize import tokenize


def _get_peak_rss_mb() -> float:
    try:
        import psutil  # type: ignore[import-not-found]

        proc = psutil.Process(os.getpid())
        info = proc.memory_info()
        peak = getattr(info, "peak_wset", info.rss)
        return float(peak) / (1024 * 1024)
    except ImportError:
        return 0.0


def run_index_benchmark(
    corpus: Path,
    executors: list[ExecutorType],
    runs: int = 3,
) -> list[dict[str, Any]]:
    cpu_count = os.cpu_count() or 4
    worker_levels = sorted({2, 4, 8, cpu_count})

    configs: list[tuple[ExecutorType, int]] = [("serial", 1)]
    for ex in executors:
        if ex == "serial":
            continue
        for w in worker_levels:
            configs.append((ex, w))

    # Warmup run
    _ = build_index(corpus, executor="serial", workers=1)

    rows: list[dict[str, Any]] = []
    serial_wall = 0.0

    for executor, workers in configs:
        wall_samples: list[float] = []
        cpu_samples: list[float] = []
        merge_samples: list[float] = []
        peak_rss_mb = 0.0

        for _ in range(runs):
            timings: BuildTiming = {"map_seconds": 0.0, "merge_seconds": 0.0, "total_seconds": 0.0}
            cpu_t0 = time.process_time()
            wall_t0 = time.perf_counter()
            _ = build_index(
                corpus,
                executor=executor,
                workers=workers,
                timings_out=timings,
            )
            wall_dt = time.perf_counter() - wall_t0
            cpu_dt = time.process_time() - cpu_t0

            wall_samples.append(wall_dt)
            cpu_samples.append(cpu_dt)
            merge_samples.append(timings["merge_seconds"])
            peak_rss_mb = max(peak_rss_mb, _get_peak_rss_mb())

        med_wall = statistics.median(wall_samples)
        med_cpu = statistics.median(cpu_samples)
        med_merge = statistics.median(merge_samples)
        if executor == "serial":
            serial_wall = med_wall
        speedup = serial_wall / med_wall if med_wall > 0 else 1.0

        row: dict[str, Any] = {
            "executor": executor,
            "workers": workers,
            "wall": round(med_wall, 4),
            "cpu": round(med_cpu, 4),
            "rss_mb": round(peak_rss_mb, 1),
            "merge": round(med_merge, 4),
            "speedup": round(speedup, 2),
        }
        rows.append(row)
        print(
            f"{executor:<10} | workers={workers:>2} | wall={med_wall:>6.3f}s | "
            f"cpu={med_cpu:>6.3f}s | rss={peak_rss_mb:>6.1f}MB | "
            f"merge={med_merge:>6.4f}s | speedup={speedup:>5.2f}x",
            flush=True,
        )

    return rows


def run_io_counter_benchmark(corpus: Path, runs: int = 3) -> list[tuple[int, float, float]]:
    refs = collect_doc_refs(corpus)
    unique_paths = sorted({Path(r.path_str) for r in refs})

    def _read_bytes(p: Path) -> int:
        return len(p.read_bytes())

    _ = sum(_read_bytes(p) for p in unique_paths)

    results: list[tuple[int, float, float]] = []
    base_wall = 0.0
    for w in [1, 2, 4, 8]:
        samples: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            if w == 1:
                _ = sum(_read_bytes(p) for p in unique_paths)
            else:
                with ThreadPoolExecutor(max_workers=w) as pool:
                    _ = sum(pool.map(_read_bytes, unique_paths))
            samples.append(time.perf_counter() - t0)
        med = statistics.median(samples)
        if w == 1:
            base_wall = med
        results.append((w, med, base_wall / med if med > 0 else 1.0))
    return results


def demonstrate_free_threaded_race(corpus: Path) -> dict[str, int]:
    """Demonstrate a shared-state race when GIL is disabled vs safe Lock / map-reduce."""
    refs = collect_doc_refs(corpus, limit=200)
    texts = [Path(r.path_str).read_text(encoding="utf-8", errors="replace") for r in refs]
    expected_total = sum(sum(1 for _ in tokenize(t)) for t in texts)

    # 1. Unsafe shared counter across 8 threads (race condition without GIL!)
    unsafe_total = 0
    unsafe_vocab: Counter[str] = Counter()

    def _unsafe_worker(chunk: list[str]) -> None:
        nonlocal unsafe_total
        for text in chunk:
            for tok in tokenize(text):
                unsafe_total += 1
                unsafe_vocab[tok] = unsafe_vocab[tok] + 1

    chunks = [texts[i::8] for i in range(8)]
    threads = [threading.Thread(target=_unsafe_worker, args=(c,)) for c in chunks]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 2. Safe version with Lock
    safe_total = 0
    lock = threading.Lock()

    def _safe_worker(chunk: list[str]) -> None:
        nonlocal safe_total
        local_count = 0
        for text in chunk:
            for _ in tokenize(text):
                local_count += 1
        with lock:
            safe_total += local_count

    threads_safe = [threading.Thread(target=_safe_worker, args=(c,)) for c in chunks]
    for t in threads_safe:
        t.start()
    for t in threads_safe:
        t.join()

    return {
        "expected": expected_total,
        "unsafe_total": unsafe_total,
        "unsafe_vocab_sum": sum(unsafe_vocab.values()),
        "safe_total": safe_total,
    }


def plot_combined_speedup(
    rows_312: list[dict[str, Any]],
    rows_313t: list[dict[str, Any]],
    out_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=150)

    serial_312 = next(r for r in rows_312 if r["executor"] == "serial")
    thread_312 = [r for r in rows_312 if r["executor"] == "threads"]
    proc_312 = [r for r in rows_312 if r["executor"] == "processes"]

    serial_313t = next(r for r in rows_313t if r["executor"] == "serial")
    thread_313t = [r for r in rows_313t if r["executor"] == "threads"]

    all_w = sorted({1} | {int(r["workers"]) for r in rows_312})
    ax.plot(all_w, all_w, linestyle="--", color="#888888", label="Ideal linear speedup (y = x)")

    t312_x = [1] + [int(r["workers"]) for r in thread_312]
    t312_y = [float(serial_312["speedup"])] + [float(r["speedup"]) for r in thread_312]
    ax.plot(
        t312_x,
        t312_y,
        marker="o",
        linewidth=2,
        color="#ff7f0e",
        label="threads (CPython 3.12, GIL=True)",
    )

    p312_x = [1] + [int(r["workers"]) for r in proc_312]
    p312_y = [float(serial_312["speedup"])] + [float(r["speedup"]) for r in proc_312]
    ax.plot(
        p312_x,
        p312_y,
        marker="s",
        linewidth=2,
        color="#1f77b4",
        label="processes (CPython 3.12, multi-core)",
    )

    # Normalize 3.13t against its own serial baseline AND show scaling
    t313_x = [1] + [int(r["workers"]) for r in thread_313t]
    t313_y = [float(serial_313t["speedup"])] + [float(r["speedup"]) for r in thread_313t]
    ax.plot(
        t313_x,
        t313_y,
        marker="^",
        linewidth=2.2,
        color="#2ca02c",
        label="threads (CPython 3.13t free-threaded, GIL=False)",
    )

    ax.set_title(
        "findex Parallel Indexing Speedup: GIL Threads vs Processes vs Free-Threaded 3.13t"
    )
    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Speedup (relative to serial = 1.0x)")
    ax.set_xticks(all_w)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["312", "313t", "plot"], required=True)
    args = parser.parse_args()

    corpus = Path("data")
    gil_on = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(
        f"Python: {sys.version.split()[0]} | GIL enabled: {gil_on} | CPU: {platform.processor()} ({os.cpu_count()} cores)"
    )

    if args.mode == "312":
        rows = run_index_benchmark(corpus, executors=["threads", "processes"], runs=3)
        io_rows = run_io_counter_benchmark(corpus, runs=3)
        payload = {"python": sys.version, "gil_enabled": gil_on, "rows": rows, "io_rows": io_rows}
        Path("docs/bench_312.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("\n=== Pure I/O Counter-Benchmark ===")
        for w, wall, sp in io_rows:
            print(
                f"threads (I/O) | workers={w:>2} | wall={wall * 1000:>6.2f} ms | speedup={sp:>5.2f}x"
            )
    elif args.mode == "313t":
        rows = run_index_benchmark(corpus, executors=["threads"], runs=3)
        race = demonstrate_free_threaded_race(corpus)
        print("\n=== Free-Threaded Data Race Check (200 docs, 8 threads) ===")
        print(f"Expected tokens:          {race['expected']:,}")
        print(
            f"Unsafe shared += 1:       {race['unsafe_total']:,} (lost {race['expected'] - race['unsafe_total']:,} updates!)"
        )
        print(
            f"Unsafe shared Counter:    {race['unsafe_vocab_sum']:,} (lost {race['expected'] - race['unsafe_vocab_sum']:,} updates!)"
        )
        print(f"Safe with threading.Lock: {race['safe_total']:,}")
        payload = {"python": sys.version, "gil_enabled": gil_on, "rows": rows, "race": race}
        Path("docs/bench_313t.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif args.mode == "plot":
        d312 = json.loads(Path("docs/bench_312.json").read_text(encoding="utf-8"))
        d313 = json.loads(Path("docs/bench_313t.json").read_text(encoding="utf-8"))
        plot_combined_speedup(d312["rows"], d313["rows"], Path("docs/speedup_lab05.png"))
        print("Saved combined plot to docs/speedup_lab05.png")


if __name__ == "__main__":
    main()
