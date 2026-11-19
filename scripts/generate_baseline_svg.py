from __future__ import annotations

import cProfile
import html
import json
import pstats
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from findex.index import build_index
from findex.search import SearchEngine


def build_corpus(path: Path, n: int = 5000) -> None:
    vocab = [
        "python",
        "asyncio",
        "event",
        "loop",
        "coroutine",
        "fastapi",
        "pydantic",
        "bm25",
        "tfidf",
        "inverted",
        "index",
        "postings",
        "ranking",
        "query",
        "multiprocessing",
        "gil",
        "threadpool",
        "numpy",
        "vector",
        "embedding",
    ]
    lines = [
        json.dumps(
            {
                "id": f"doc-{i}",
                "title": f"Doc {i}",
                "text": " ".join(vocab[(i + j * 7) % len(vocab)] for j in range(60)),
            }
        )
        for i in range(n)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_flame_svg(stats: pstats.Stats, out_path: Path, title: str) -> None:
    rows: list[tuple[str, float, float, int]] = []
    total_time = 0.0
    for (filename, line, func_name), (_cc, nc, tt, ct, _callers) in stats.stats.items():  # type: ignore[attr-defined]
        total_time += tt
        short_file = Path(filename).name
        rows.append((f"{short_file}:{line} ({func_name})", tt, ct, nc))
    rows.sort(key=lambda r: r[2], reverse=True)
    top_rows = rows[:18]
    max_ct = max((r[2] for r in top_rows), default=1.0) or 1.0

    width = 1100
    bar_h = 26
    pad = 16
    height = 70 + len(top_rows) * (bar_h + 6)
    palette = ["#e05a47", "#e87a3d", "#f0a035", "#d9534f", "#c96a3b", "#b87d35"]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#181a1f"/>',
        f'<text x="{pad}" y="32" fill="#f3f4f6" font-family="Consolas, monospace" font-size="16" font-weight="bold">{html.escape(title)} (total={total_time:.2f}s)</text>',
    ]
    for idx, (label, tt, ct, nc) in enumerate(top_rows):
        y = 52 + idx * (bar_h + 6)
        bar_w = max(8.0, (ct / max_ct) * (width - 2 * pad))
        color = (
            palette[idx % len(palette)]
            if "findex" in label or "ranking.py" in label or "index.py" in label
            else "#3b6ea5"
        )
        pct = (ct / max(total_time, 1e-9)) * 100.0
        text = f"{label} — cum={ct:.3f}s ({pct:.1f}%), self={tt:.3f}s, calls={nc:,}"
        svg_parts.append(
            f'<rect x="{pad}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="4" fill="{color}"/>'
        )
        svg_parts.append(
            f'<text x="{pad + 8}" y="{y + 17}" fill="#ffffff" font-family="Consolas, monospace" font-size="12">{html.escape(text)}</text>'
        )
    svg_parts.append("</svg>")
    out_path.write_text("\n".join(svg_parts), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus_5k.jsonl"
        build_corpus(corpus, n=5000)
        idx = build_index(corpus, executor="serial")
        engine = SearchEngine(idx)

        pr = cProfile.Profile()
        pr.enable()
        for i in range(250):
            engine._compute_ranked(f"python OR bm25 OR numpy OR doc_{i}", "bm25", 10)
        pr.disable()

        render_flame_svg(
            pstats.Stats(pr),
            _ROOT / "docs" / "search_baseline.svg",
            "Baseline Search Profile (Lab 03 Scalar Scorer, 250 queries x 5,000 docs)",
        )


if __name__ == "__main__":
    main()
