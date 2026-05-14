"""
Centralised filesystem-path utilities with corpus-aware resolution.

This is the only module that resolves project paths. Every script imports
its directories from here so there are no hard-coded paths anywhere else in
the codebase. Moving a directory is a one-line change in this file.

Layout::

    data/
      metadata/                       # taxonomies (corpus-independent, committed)
      corpora/
        public/                       # synthesised dossiers (committed raw, derived gitignored)
          raw/                        # source PDFs
          parsed_text/                # parsed text + metadata sidecars
          chunks/                     # chunk-level artefacts
          knowledge/                  # retrieval-ready knowledge base
          vector_store/               # embeddings + FAISS index
        private/                      # full-book PDFs (local-only, fully gitignored)
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


def get_evaluation_dir() -> Path:
    """Return ``<root>/evaluation`` - evaluation harness inputs and outputs."""
    return get_project_root() / "evaluation"


def get_response_dir() -> Path:
    """Return ``<root>/response`` — timestamped eval JSON, ``connectivity_log.txt``, ``sample_results_comparison.md``."""
    return get_project_root() / "response"


def ensure_directory(path: Path | str) -> Path:
    """Create the directory if it does not exist and return it as a ``Path``.

    Used at the top of pipeline scripts so first-run executions do not fail
    with ``FileNotFoundError`` when intermediate output directories are absent.
    """

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
