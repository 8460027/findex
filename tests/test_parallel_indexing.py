from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from findex.index import DocRef, build_index, build_partial, collect_doc_refs, merge
from findex.store import save_index


@pytest.mark.parametrize("with_positions", [False, True])
def test_serial_threads_processes_produce_identical_indexes(
    sample_corpus_dir: Path, tmp_path: Path, with_positions: bool
) -> None:
    serial_idx = build_index(
        sample_corpus_dir, with_positions=with_positions, executor="serial", workers=1
    )
    threads_idx = build_index(
        sample_corpus_dir, with_positions=with_positions, executor="threads", workers=3
    )
    procs_idx = build_index(
        sample_corpus_dir, with_positions=with_positions, executor="processes", workers=3
    )

    assert dict(threads_idx.postings) == dict(serial_idx.postings)
    assert dict(procs_idx.postings) == dict(serial_idx.postings)
    assert threads_idx.doc_lengths == serial_idx.doc_lengths
    assert procs_idx.doc_lengths == serial_idx.doc_lengths
    assert threads_idx.doc_meta == serial_idx.doc_meta
    assert procs_idx.doc_meta == serial_idx.doc_meta

    # Verify byte-identical serialized binary output
    b_serial = tmp_path / f"serial_{with_positions}.bin"
    b_threads = tmp_path / f"threads_{with_positions}.bin"
    b_procs = tmp_path / f"procs_{with_positions}.bin"
    save_index(serial_idx, b_serial, fmt="binary")
    save_index(threads_idx, b_threads, fmt="binary")
    save_index(procs_idx, b_procs, fmt="binary")

    assert b_threads.read_bytes() == b_serial.read_bytes()
    assert b_procs.read_bytes() == b_serial.read_bytes()


def test_map_reduce_split_matches_lab04_single_pass(sample_corpus_dir: Path) -> None:
    refs = collect_doc_refs(sample_corpus_dir)
    mid = len(refs) // 2
    p1 = build_partial(refs[:mid], with_positions=True)
    p2 = build_partial(refs[mid:], with_positions=True)
    merged = merge([p1, p2])
    baseline = build_index(sample_corpus_dir, with_positions=True, executor="serial")

    assert dict(merged.postings) == dict(baseline.postings)
    assert merged.doc_lengths == baseline.doc_lengths


def _failing_worker(doc_refs: list[DocRef], *, with_positions: bool = False) -> None:
    raise RuntimeError("Simulated worker crash")


def test_worker_exception_surfaces_loudly(sample_corpus_dir: Path) -> None:
    with (
        patch("findex.index.build_partial", side_effect=_failing_worker),
        pytest.raises(RuntimeError, match="Simulated worker crash"),
    ):
        build_index(sample_corpus_dir, executor="threads", workers=2)
