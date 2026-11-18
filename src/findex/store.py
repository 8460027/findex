import argparse
import json
import pickle
import struct
import time
from array import array
from pathlib import Path

from findex.models import DocMeta, InvertedIndex, Posting

# SECURITY WARNING:
# pickle.load() executes arbitrary Python bytecode embedded in the stream.
# Never call load_pickle() on files downloaded from untrusted sources or over a network.
# Use it strictly for index files generated locally by your own process.


def save_pickle(obj: object, path: Path) -> float:
    t0 = time.perf_counter()
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return time.perf_counter() - t0


def load_pickle(path: Path) -> object:
    with path.open("rb") as f:
        return pickle.load(f)


def save_json(index: InvertedIndex, path: Path) -> float:
    t0 = time.perf_counter()
    payload = {
        "with_positions": index.with_positions,
        "doc_lengths": index.doc_lengths,
        "doc_meta": {
            str(k): {"path": v.path, "title": v.title}
            for k, v in index.doc_meta.items()
        },
        "postings": {
            term: [
                [p.doc_id, p.tf, list(p.positions)] if index.with_positions else [p.doc_id, p.tf]
                for p in plist
            ]
            for term, plist in index.postings.items()
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    return time.perf_counter() - t0


def load_json(path: Path) -> InvertedIndex:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    with_positions = payload["with_positions"]
    doc_lengths = {int(k): v for k, v in payload["doc_lengths"].items()}
    doc_meta = {
        int(k): DocMeta(doc_id=int(k), path=v["path"], title=v["title"])
        for k, v in payload["doc_meta"].items()
    }
    postings = {}
    for term, plist in payload["postings"].items():
        if with_positions:
            postings[term] = [Posting(p[0], p[1], tuple(p[2])) for p in plist]
        else:
            postings[term] = [Posting(p[0], p[1]) for p in plist]
    return InvertedIndex(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        with_positions=with_positions,
    )


def save_binary(index: InvertedIndex, path: Path) -> float:
    """Compact binary format: [8-byte header_len][JSON header][contiguous uint32 arrays]."""
    t0 = time.perf_counter()
    term_offsets: dict[str, tuple[int, int]] = {}
    blob = bytearray()

    for term, plist in index.postings.items():
        offset = len(blob)
        count = len(plist)
        arr = array("I")
        for p in plist:
            arr.append(p.doc_id)
            arr.append(p.tf)
        blob.extend(arr.tobytes())
        term_offsets[term] = (offset, count)

    header = {
        "doc_lengths": index.doc_lengths,
        "doc_meta": {
            str(k): {"path": v.path, "title": v.title}
            for k, v in index.doc_meta.items()
        },
        "terms": term_offsets,
    }
    header_bytes = json.dumps(header).encode("utf-8")

    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(blob)

    return time.perf_counter() - t0


def load_binary(path: Path) -> InvertedIndex:
    raw = path.read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    data_view = memoryview(raw)[8 + header_len :]

    doc_lengths = {int(k): v for k, v in header["doc_lengths"].items()}
    doc_meta = {
        int(k): DocMeta(doc_id=int(k), path=v["path"], title=v["title"])
        for k, v in header["doc_meta"].items()
    }
    postings: dict[str, list[Posting]] = {}
    for term, (offset, count) in header["terms"].items():
        arr = array("I")
        arr.frombytes(data_view[offset : offset + count * 8])
        postings[term] = [
            Posting(arr[i], arr[i + 1]) for i in range(0, len(arr), 2)
        ]

    return InvertedIndex(
        postings=postings,
        doc_lengths=doc_lengths,
        doc_meta=doc_meta,
        with_positions=False,
    )


def save_index(index: InvertedIndex, path: Path, fmt: str = "pickle") -> float:
    if fmt == "pickle":
        return save_pickle(index, path)
    if fmt == "json":
        return save_json(index, path)
    if fmt == "binary":
        return save_binary(index, path)
    raise ValueError(f"Unknown format: {fmt}")


def load_index(path: Path, fmt: str = "pickle") -> InvertedIndex:
    if fmt == "pickle":
        loaded = load_pickle(path)
        assert isinstance(loaded, InvertedIndex)
        return loaded
    if fmt == "json":
        return load_json(path)
    if fmt == "binary":
        return load_binary(path)
    raise ValueError(f"Unknown format: {fmt}")


def benchmark_formats(index: InvertedIndex) -> None:
    tmp_dir = Path(".bench_tmp")
    tmp_dir.mkdir(exist_ok=True)

    print("\n| Format | File size | Save time | Load time |")
    print("|---|---|---|---|")
    for fmt, ext in [("pickle", "pkl"), ("json", "json"), ("binary", "bin")]:
        out_path = tmp_dir / f"idx.{ext}"
        st = save_index(index, out_path, fmt=fmt)
        t0 = time.perf_counter()
        _ = load_index(out_path, fmt=fmt)
        lt = time.perf_counter() - t0
        sz = out_path.stat().st_size / 1e6
        print(f"| `{fmt}` | {sz:.2f} MB | {st:.4f} s | {lt:.4f} s |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark serialization formats.")
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()
    from findex.index import build_index

    idx = build_index(args.corpus)
    benchmark_formats(idx)

