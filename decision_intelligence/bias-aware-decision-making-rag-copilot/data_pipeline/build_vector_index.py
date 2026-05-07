"""
Build embeddings and a FAISS index from the enriched knowledge base.
"""

from __future__ import annotations

import json

import numpy as np

from pathlib import Path

try:
    from data_pipeline.bootstrap import add_project_root_to_sys_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_pipeline.bootstrap import add_project_root_to_sys_path

add_project_root_to_sys_path()

from app.services.bedrock_provider import BedrockProvider
from shared_components.utilities.path_utils import (
    ensure_directory,
    get_knowledge_dir,
    get_vector_store_dir,
)


INPUT_FILE = get_knowledge_dir() / "knowledge_base.json"
OUTPUT_DIR = ensure_directory(get_vector_store_dir())
INDEX_FILE = OUTPUT_DIR / "knowledge.index"
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "index_metadata.json"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"


def build_index():
    provider = BedrockProvider()
    try:
        import faiss
    except ImportError:
        faiss = None

    with INPUT_FILE.open("r", encoding="utf-8") as handle:
        knowledge_base = json.load(handle)

    print(f"Generating embeddings for {len(knowledge_base)} chunks...")
    embeddings = np.asarray(
        [provider.embed_text(chunk["text"]) for chunk in knowledge_base],
        dtype=np.float32,
    )

    dimension = embeddings.shape[1]
    np.save(EMBEDDINGS_FILE, embeddings)

    index_type = "numpy_fallback"
    if faiss is not None:
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        faiss.write_index(index, str(INDEX_FILE))
        index_type = "IndexFlatIP"

    with METADATA_FILE.open("w", encoding="utf-8") as handle:
        json.dump(knowledge_base, handle, indent=2)

    with MANIFEST_FILE.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "count": len(knowledge_base),
                "dimension": dimension,
                "index_type": index_type,
            },
            handle,
            indent=2,
        )

    if faiss is not None:
        print(f"\n✅ Vector index created at:\n{INDEX_FILE}")
    else:
        print("\n⚠️ FAISS is not installed. Embeddings and metadata were created using numpy-only fallback.")


if __name__ == "__main__":
    build_index()
