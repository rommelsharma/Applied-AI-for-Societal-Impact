"""Build ``processed/index/bm25_index.pkl`` from ``knowledge_base.json``."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

try:
    from data_pipeline.bootstrap import add_project_root_to_sys_path
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_pipeline.bootstrap import add_project_root_to_sys_path

add_project_root_to_sys_path()

from rank_bm25 import BM25Okapi

from shared_components.utilities.path_utils import (
    ensure_directory,
    get_corpus_name,
    get_processed_index_dir,
    resolve_knowledge_base_json,
)
from shared_components.utilities.taxonomy_utils import get_ontology_version


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


def build_bm25(corpus: str | None = None) -> Path:
    resolved = get_corpus_name(corpus)
    kb_path = resolve_knowledge_base_json(resolved)
    with kb_path.open("r", encoding="utf-8") as handle:
        kb = json.load(handle)

    tokenized_docs = [tokenize(row.get("text", "")) for row in kb if isinstance(row, dict)]
    tokenized_docs = [d if d else ["empty"] for d in tokenized_docs]

    index_dir = ensure_directory(get_processed_index_dir(resolved))
    out_path = index_dir / "bm25_index.pkl"
    payload = {
        "version": 1,
        "corpus": resolved,
        "ontology_version": get_ontology_version(),
        "tokenized_docs": tokenized_docs,
    }
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    bm25 = BM25Okapi(tokenized_docs)
    _ = bm25.get_scores(tokenize("anchoring bias hiring"))
    print(f"BM25 index saved ({len(tokenized_docs)} docs) → {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=None)
    return p.parse_args()


if __name__ == "__main__":
    build_bm25(corpus=parse_args().corpus)
