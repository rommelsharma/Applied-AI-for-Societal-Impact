"""
Vector-index builder - step 5 of the offline pipeline.

Embeds every record in ``<corpus>/knowledge/knowledge_base.json`` using
Amazon Titan Text Embeddings v2 and persists four aligned artefacts in
``<corpus>/vector_store/``:

    * ``embeddings.npy``       - the raw float32 matrix (rows aligned with the index)
    * ``knowledge.index``      - a FAISS ``IndexFlatIP`` over the same vectors
    * ``index_metadata.json``  - the enriched chunk payload, in matching row order
    * ``manifest.json``        - small summary including count / dimension / index type
                                 plus the corpus name and chunking-profile mix

Why ``IndexFlatIP`` with normalised embeddings:
    * FAISS inner-product search on unit vectors == cosine similarity.
    * Flat indexes are exact (zero approximation error) which matters for a
      small, high-quality corpus.
    * The retriever depends only on the ``search(query_vector, k)`` interface
      so this index can later be swapped for ``IndexIVFFlat`` or HNSW
      without touching ``rag/retriever.py``.

A pure-numpy fallback is preserved for environments without FAISS installed.

Two CLI affordances support fast iteration:
    * ``--corpus public|private|...`` selects the corpus.
    * ``--metadata-only`` refreshes only ``index_metadata.json`` from the
      enriched knowledge base, without re-embedding. Use this after adding
      passage_type / decision_phase tags or fixing enrichment fields when
      the underlying chunk text has not changed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

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
    get_corpus_name,
    get_knowledge_dir,
    get_vector_store_dir,
)


def _resolve_paths(corpus: str | None):
    """Return ``(input_file, output_dir, individual artefact paths)`` for a corpus."""
    input_file = get_knowledge_dir(corpus) / "knowledge_base.json"
    output_dir = ensure_directory(get_vector_store_dir(corpus))
    return {
        "input_file": input_file,
        "output_dir": output_dir,
        "index_file": output_dir / "knowledge.index",
        "embeddings_file": output_dir / "embeddings.npy",
        "metadata_file": output_dir / "index_metadata.json",
        "manifest_file": output_dir / "manifest.json",
    }


def build_index(corpus: str | None = None, metadata_only: bool = False):
    """Embed every knowledge-base record (or refresh metadata only) for a corpus.

    Args:
        corpus: Corpus name override; defaults to ``CORPUS_NAME`` env var.
        metadata_only: When ``True`` skip embedding and only refresh
            ``index_metadata.json`` from the current knowledge base. Use
            this when chunk *text* is unchanged but enrichment fields
            (passage_type, decision_phase, etc.) have been updated.
    """
    paths = _resolve_paths(corpus)
    resolved_corpus = get_corpus_name(corpus)

    try:
        import faiss
    except ImportError:
        faiss = None

    with paths["input_file"].open("r", encoding="utf-8") as handle:
        knowledge_base = json.load(handle)

    profile_distribution = Counter(chunk.get("chunking_profile", "unknown") for chunk in knowledge_base)

    if metadata_only:
        # Refresh just the alignment metadata. Useful after re-running
        # enrichment on already-embedded chunks.
        with paths["metadata_file"].open("w", encoding="utf-8") as handle:
            json.dump(knowledge_base, handle, indent=2)
        manifest = {
            "corpus": resolved_corpus,
            "count": len(knowledge_base),
            "dimension": _detect_dimension_from_existing(paths),
            "index_type": _detect_index_type(paths, faiss is not None),
            "chunking_profile_distribution": dict(profile_distribution),
            "metadata_only_refresh": True,
        }
        with paths["manifest_file"].open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        print(f"\n✅ Metadata-only refresh complete for corpus '{resolved_corpus}'.")
        return

    provider = BedrockProvider()

    print(f"Generating embeddings for {len(knowledge_base)} chunks (corpus: {resolved_corpus})...")
    # Each call hits Bedrock; this is the most expensive part of the build
    # and the reason this script is run independently of the rest of the pipeline.
    embeddings = np.asarray(
        [provider.embed_text(chunk["text"]) for chunk in knowledge_base],
        dtype=np.float32,
    )

    dimension = embeddings.shape[1]
    np.save(paths["embeddings_file"], embeddings)

    index_type = "numpy_fallback"
    if faiss is not None:
        # IndexFlatIP + normalised embeddings == exact cosine similarity. No
        # training needed for flat indexes, so ``add`` is the only call required.
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        faiss.write_index(index, str(paths["index_file"]))
        index_type = "IndexFlatIP"

    with paths["metadata_file"].open("w", encoding="utf-8") as handle:
        json.dump(knowledge_base, handle, indent=2)

    with paths["manifest_file"].open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "corpus": resolved_corpus,
                "count": len(knowledge_base),
                "dimension": dimension,
                "index_type": index_type,
                "chunking_profile_distribution": dict(profile_distribution),
            },
            handle,
            indent=2,
        )

    if faiss is not None:
        print(f"\n✅ Vector index created at:\n{paths['index_file']}")
    else:
        print("\n⚠️ FAISS is not installed. Embeddings and metadata were created using numpy-only fallback.")


def _detect_dimension_from_existing(paths) -> int | None:
    """Read the dimension from the existing embeddings.npy without re-embedding."""
    if paths["embeddings_file"].exists():
        return int(np.load(paths["embeddings_file"]).shape[1])
    return None


def _detect_index_type(paths, faiss_available: bool) -> str:
    """Report which backend is actually persisted on disk for this corpus."""
    if faiss_available and paths["index_file"].exists():
        return "IndexFlatIP"
    if paths["embeddings_file"].exists():
        return "numpy_fallback"
    return "unbuilt"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. ``--corpus`` and ``--metadata-only`` are optional."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=None, help="Corpus to build (defaults to CORPUS_NAME or 'public').")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip embedding; refresh only index_metadata.json from the current knowledge base.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(corpus=args.corpus, metadata_only=args.metadata_only)
