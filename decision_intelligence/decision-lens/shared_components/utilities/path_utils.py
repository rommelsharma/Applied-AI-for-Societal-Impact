"""
Centralised filesystem-path utilities with corpus-aware resolution.

This is the only module that resolves project paths. Every script imports
its directories from here so there are no hard-coded paths anywhere else in
the codebase. Moving a directory is a one-line change in this file.

Layout::

    data/
      registry/                       # decision_intelligence_ontology.json
      eval/
        gold/                         # frozen ``scenarios_catalog.json`` (canonical eval scenarios)
        runs/                         # connectivity_log, eval JSON, sample_results JSONL
      metadata/                       # legacy bias-taxonomy (LLM closed vocabulary)
      corpora/
        public/
          raw/  parsed_text/  chunks/
          knowledge/                  # legacy KB path (optional)
          vector_store/               # legacy index (optional)
          processed/
            enriched/                 # knowledge_base.json, synthesis.json
            index/                    # FAISS, embeddings.npy, bm25_index.pkl
        private/
          raw/  ...

Every path resolver accepts an optional ``corpus`` argument so a single
process can address two corpora at once (used by ``compare_versions.py``).
When ``corpus`` is omitted, the value of the ``CORPUS_NAME`` environment
variable wins, falling back to ``"public"``.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path of the project root.

    Resolves relative to this file's location, so the answer is correct
    regardless of the caller's working directory.
    """

    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """Return ``<root>/data`` - the parent of all corpus artefacts."""
    return get_project_root() / "data"


def get_metadata_dir() -> Path:
    """Return ``<root>/data/metadata`` - taxonomies and dossier manifests.

    Corpus-independent: shared across every corpus version.
    """
    return get_data_dir() / "metadata"


def get_registry_dir() -> Path:
    """Return ``<root>/data/registry`` — ontology brain (``decision_intelligence_ontology.json``)."""
    return get_data_dir() / "registry"


def get_eval_gold_dir() -> Path:
    """Frozen scenarios and gold labels (``data/eval/gold``)."""
    return get_data_dir() / "eval" / "gold"


def get_processed_root(corpus: str | None = None) -> Path:
    """Return ``<corpus>/processed`` for v4-style artefact layout."""
    return get_corpus_dir(corpus) / "processed"


def get_processed_enriched_dir(corpus: str | None = None) -> Path:
    """``<corpus>/processed/enriched`` — ``knowledge_base.json``, ``synthesis.json``."""
    return get_processed_root(corpus) / "enriched"


def get_processed_index_dir(corpus: str | None = None) -> Path:
    """``<corpus>/processed/index`` — FAISS, embeddings, BM25 pickle."""
    return get_processed_root(corpus) / "index"


def resolve_knowledge_base_json(corpus: str | None = None) -> Path:
    """Prefer ``processed/enriched/knowledge_base.json``, else legacy ``knowledge/knowledge_base.json``."""
    new_p = get_processed_enriched_dir(corpus) / "knowledge_base.json"
    if new_p.is_file():
        return new_p
    return get_knowledge_dir(corpus) / "knowledge_base.json"


def resolve_vector_index_dir(corpus: str | None = None) -> Path:
    """Directory with ``knowledge.index`` / ``embeddings.npy`` — prefer ``processed/index``, else ``vector_store``."""
    new_d = get_processed_index_dir(corpus)
    old_d = get_vector_store_dir(corpus)

    def _has_index(d: Path) -> bool:
        return (d / "knowledge.index").is_file() or (d / "embeddings.npy").is_file()

    if _has_index(new_d):
        return new_d
    if _has_index(old_d):
        return old_d
    return new_d


def resolve_synthesis_json(corpus: str | None = None) -> Path:
    """Path to ``synthesis.json`` (may not exist until synthesis_builder runs)."""
    return get_processed_enriched_dir(corpus) / "synthesis.json"


def get_corpora_root() -> Path:
    """Return ``<root>/data/corpora`` - the parent of all named corpora."""
    return get_data_dir() / "corpora"


def get_corpus_name(corpus: str | None = None) -> str:
    """Resolve the corpus name from argument or environment.

    Defaults to ``"public"`` so unconfigured local runs use the committed
    dossier corpus.
    """
    return corpus or os.getenv("CORPUS_NAME", "public")


def get_corpus_dir(corpus: str | None = None) -> Path:
    """Return ``<root>/data/corpora/<corpus>``."""
    return get_corpora_root() / get_corpus_name(corpus)


def get_raw_data_dir(corpus: str | None = None) -> Path:
    """Return ``<corpus>/raw`` - source PDFs for the chosen corpus."""
    return get_corpus_dir(corpus) / "raw"


def get_parsed_text_dir(corpus: str | None = None) -> Path:
    """Return ``<corpus>/parsed_text`` - per-PDF text + metadata sidecars."""
    return get_corpus_dir(corpus) / "parsed_text"


def get_chunks_dir(corpus: str | None = None) -> Path:
    """Return ``<corpus>/chunks`` - chunk-level artefacts before enrichment."""
    return get_corpus_dir(corpus) / "chunks"


def get_knowledge_dir(corpus: str | None = None) -> Path:
    """Return ``<corpus>/knowledge`` - the final retrieval-ready knowledge base."""
    return get_corpus_dir(corpus) / "knowledge"


def get_vector_store_dir(corpus: str | None = None) -> Path:
    """Return ``<corpus>/vector_store`` - embeddings, FAISS index, manifest."""
    return get_corpus_dir(corpus) / "vector_store"


def get_prompts_dir() -> Path:
    """Return ``<root>/prompts`` - LLM prompt templates."""
    return get_project_root() / "prompts"


def get_bias_detection_system_prompt_path() -> Path:
    """Return the versioned system prompt for bias detection (currently ``prompts/v1/``)."""
    return get_prompts_dir() / "v1" / "bias_detection_system_prompt.txt"


def get_evaluation_dir() -> Path:
    """Return ``<root>/evaluation`` - evaluation harness inputs and outputs."""
    return get_project_root() / "evaluation"


def get_response_dir() -> Path:
    """Return ``<root>/data/eval/runs`` — append-only eval captures (v4 layout).

    Legacy ``<root>/response`` is still migrated/read by ``evaluation.sample_results_log``
    when migrating old logs.
    """
    return get_data_dir() / "eval" / "runs"


def get_legacy_response_dir() -> Path:
    """Pre-v4 top-level ``response/`` directory (removed from repo layout).

    Kept so ``migrate_legacy_response_dir`` can copy stray files if they exist
    locally. Canonical eval captures use :func:`get_response_dir` (``data/eval/runs``).
    """
    return get_project_root() / "response"


def ensure_directory(path: Path | str) -> Path:
    """Create the directory if it does not exist and return it as a ``Path``.

    Used at the top of pipeline scripts so first-run executions do not fail
    with ``FileNotFoundError`` when intermediate output directories are absent.
    """

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
