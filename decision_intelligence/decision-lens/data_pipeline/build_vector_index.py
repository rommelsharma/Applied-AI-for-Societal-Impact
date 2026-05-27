"""
Vector-index builder - step 5 of the offline pipeline.

Embeds every record in ``<corpus>/processed/enriched/knowledge_base.json`` (or
legacy ``<corpus>/knowledge/knowledge_base.json``) using Amazon Titan Text
Embeddings v2 and persists aligned artefacts in ``<corpus>/processed/index/``
(legacy: ``<corpus>/vector_store/``):

By default each chunk is embedded using **sentence-centred windows**: for every
sentence, Titan embeds that sentence plus up to ``RAG_EMBED_SENTENCE_RADIUS``
sentences before and after (clamped at chunk edges); vectors are **mean-pooled**
and L2-normalised so each chunk still maps to **one** FAISS row aligned with
``index_metadata.json``. Set ``RAG_EMBED_SENTENCE_WINDOWS=false`` to embed raw
``chunk['text']`` as a single call. Optional ``RAG_EMBEDDING_MAX_WINDOWS`` caps
Bedrock calls per chunk when subsampling windows.

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
import re
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
from shared_components.settings import RAG_SETTINGS
from shared_components.utilities.path_utils import (
    ensure_directory,
    get_corpus_name,
    get_processed_index_dir,
    resolve_knowledge_base_json,
)
from shared_components.utilities.taxonomy_utils import get_ontology_version


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace.

    Falls back to a single segment when no boundaries are found so short or
    unpunctuated passages still embed sensibly.
    """
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = [p.strip() for p in parts if p.strip()]
    return out if out else [text]


def _sentence_centered_windows(sentences: list[str], radius: int) -> list[str]:
    """For each sentence index ``i``, join ``sentences[i-radius : i+radius+1]`` (clamped)."""
    n = len(sentences)
    windows: list[str] = []
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        windows.append(" ".join(sentences[lo:hi]))
    return windows


def _maybe_subsample_windows(windows: list[str], max_windows: int) -> list[str]:
    """Evenly subsample when ``max_windows > 0`` and len(windows) exceeds the cap."""
    if max_windows <= 0 or len(windows) <= max_windows:
        return windows
    positions = np.linspace(0, len(windows) - 1, num=max_windows)
    pick = sorted({int(round(float(p))) for p in positions})
    return [windows[i] for i in pick]


def _embedding_vector_for_chunk(provider: BedrockProvider, chunk: dict) -> list[float]:
    """Return a single L2-normalised embedding vector for one knowledge-base row.

    When ``RAG_EMBED_SENTENCE_WINDOWS`` is true, builds one text window per sentence
    (``radius`` sentences before and after, inclusive), optionally subsamples windows,
    embeds each window with Titan, then mean-pools and re-normalises so FAISS IP
    still matches cosine similarity. Otherwise embeds ``chunk['text']`` as before.
    """
    text = chunk.get("text") or ""
    if not RAG_SETTINGS.embed_sentence_windows:
        return provider.embed_text(text)

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return provider.embed_text(text)

    windows = _sentence_centered_windows(sentences, RAG_SETTINGS.embed_sentence_radius)
    windows = _maybe_subsample_windows(windows, RAG_SETTINGS.embed_max_windows_per_chunk)
    if not windows:
        return provider.embed_text(text)

    stacked = np.asarray([provider.embed_text(w) for w in windows], dtype=np.float32)
    mean_v = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean_v))
    if norm > 0:
        mean_v = mean_v / norm
    return mean_v.tolist()


def _resolve_paths(corpus: str | None):
    """Return artefact paths for a corpus (v4 ``processed/index`` layout)."""
    input_file = resolve_knowledge_base_json(corpus)
    output_dir = ensure_directory(get_processed_index_dir(corpus))
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
            "embedding_sentence_windows": RAG_SETTINGS.embed_sentence_windows,
            "embedding_sentence_radius": RAG_SETTINGS.embed_sentence_radius,
            "embedding_max_windows_per_chunk": RAG_SETTINGS.embed_max_windows_per_chunk,
            "ontology_version": get_ontology_version(),
        }
        with paths["manifest_file"].open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        print(f"\n✅ Metadata-only refresh complete for corpus '{resolved_corpus}'.")
        return

    provider = BedrockProvider()

    print(f"Generating embeddings for {len(knowledge_base)} chunks (corpus: {resolved_corpus})...")
    # Each call hits Bedrock; sentence-window mode issues multiple embed calls per
    # chunk then mean-pools (see ``_embedding_vector_for_chunk``).
    rows: list[list[float]] = []
    total = len(knowledge_base)
    for idx, chunk in enumerate(knowledge_base):
        if idx == 0 or (idx + 1) % 10 == 0 or idx == total - 1:
            print(f"  … chunk {idx + 1}/{total}", flush=True)
        rows.append(_embedding_vector_for_chunk(provider, chunk))
    embeddings = np.asarray(rows, dtype=np.float32)

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
                "embedding_sentence_windows": RAG_SETTINGS.embed_sentence_windows,
                "embedding_sentence_radius": RAG_SETTINGS.embed_sentence_radius,
                "embedding_max_windows_per_chunk": RAG_SETTINGS.embed_max_windows_per_chunk,
                "ontology_version": get_ontology_version(),
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
