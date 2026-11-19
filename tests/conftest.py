from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import json
from pathlib import Path

import pytest

from findex.index import Index, build_index
from findex.store import save_index


@pytest.fixture
def sample_corpus_dir(tmp_path: Path) -> Path:
    """Create a small multi-document corpus with varying lengths and languages."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    (corpus / "doc0.txt").write_text(
        "Python search engine inverted index with BM25 ranking.",
        encoding="utf-8",
    )
    (corpus / "doc1.txt").write_text(
        "Python generators yield items lazily in constant memory without lists.",
        encoding="utf-8",
    )
    (corpus / "doc2.md").write_text(
        "# Information Retrieval\n"
        "TF-IDF and BM25 score documents using term frequency and inverse document frequency. "
        "Python dictionaries map terms to sorted postings lists for fast Boolean retrieval.",
        encoding="utf-8",
    )
    (corpus / "doc3.txt").write_text(
        "РЁРІРёРґРєРёР№ РїРѕС€СѓРєРѕРІРёР№ СЂСѓС€С–Р№ РЅР° Python РїС–РґС‚СЂРёРјСѓС” СѓРєСЂР°С—РЅСЃСЊРєСѓ РјРѕРІСѓ, Р°РїРѕСЃС‚СЂРѕС„ Рї'СЏС‚СЊ С‚Р° РґРµС„С–СЃ.",
        encoding="utf-8",
    )
    jsonl_records = [
        {
            "id": "jsonl_1",
            "text": "Rare unicorn term appears once here for IDF sanity check.",
        },
        {
            "id": "jsonl_2",
            "text": (
                "Long document filler words repeated many times to test document length "
                "normalization in BM25 versus standard TF-IDF scoring models across large corpora "
                "with extra background narrative and unicorn mention at the end."
            ),
        },
    ]
    (corpus / "extra.jsonl").write_text(
        "\n".join(json.dumps(rec, ensure_ascii=False) for rec in jsonl_records) + "\n",
        encoding="utf-8",
    )
    return corpus


@pytest.fixture
def sample_index(sample_corpus_dir: Path) -> Index:
    """Build an in-memory Index with positions enabled."""
    return build_index(sample_corpus_dir, with_positions=True)


@pytest.fixture
def persisted_index_path(sample_index: Index, tmp_path: Path) -> Path:
    """Save sample_index in binary format and return its file path."""
    out = tmp_path / "sample.bin"
    save_index(sample_index, out, fmt="binary")
    return out
