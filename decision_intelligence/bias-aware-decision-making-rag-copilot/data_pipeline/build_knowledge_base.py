"""
Master orchestrator for the offline ingestion pipeline.

Runs the four ingestion steps in order:
    1. Parse PDFs into text + metadata sidecars.
    2. Chunk the parsed text with front/back-matter trimming and chapter detection.
    3. Tag every chunk with taxonomy concepts.
    4. Enrich each chunk into a retrieval-ready knowledge-base record (including
       passage_type / decision_phase classification).

The vector-index build (step 5) is intentionally a separate script because
it requires live Bedrock access and produces large binary artefacts, which
makes it useful to run independently of the corpus refresh.

Corpus-aware via ``--corpus`` (defaults to ``CORPUS_NAME`` env or ``public``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from data_pipeline.bootstrap import add_project_root_to_sys_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_pipeline.bootstrap import add_project_root_to_sys_path

add_project_root_to_sys_path()

from data_pipeline.chunker import create_chunks
from data_pipeline.concept_extractor import process_chunks
from data_pipeline.enrich_chunks import enrich
from data_pipeline.pdf_parser import parse_all_pdfs


def run_pipeline(
    corpus: str | None = None,
    profile: str | None = None,
    *,
    skip_parse: bool = False,
):
    """Execute parse → chunk → concept-tag → enrich in sequence.

    Set ``skip_parse=True`` to skip the PDF parsing step. This is useful when
    parsed-text has been written directly into ``parsed_text/`` by another
    ingestion path (for example ``scripts/ingest_author_content.py``) and
    re-running the PDF parser would either be wasteful or would overwrite
    those externally-produced sidecars.
    """
    print(f"\nCorpus: {corpus or '(env-default)'} | Profile: {profile or '(auto/env)'}")

    if skip_parse:
        print("\n==============================")
        print("STEP 1 — PARSING PDFs (skipped)")
        print("==============================")
    else:
        print("\n==============================")
        print("STEP 1 — PARSING PDFs")
        print("==============================")
        parse_all_pdfs(corpus=corpus)

    print("\n==============================")
    print("STEP 2 — CREATING CHUNKS")
    print("==============================")
    create_chunks(corpus=corpus, profile=profile)

    print("\n==============================")
    print("STEP 3 — EXTRACTING CONCEPTS")
    print("==============================")
    process_chunks(corpus=corpus)

    print("\n==============================")
    print("STEP 4 — ENRICHING KNOWLEDGE")
    print("==============================")
    enrich(corpus=corpus)

    print("\n✅ KNOWLEDGE BASE PIPELINE COMPLETE")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. ``--corpus``, ``--profile`` and ``--skip-parse`` are optional."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=None, help="Corpus to ingest (defaults to CORPUS_NAME or 'public').")
    parser.add_argument(
        "--profile",
        default=None,
        choices=["dossier", "book", "auto"],
        help="Chunking profile override.",
    )
    parser.add_argument(
        "--skip-parse",
        action="store_true",
        help=(
            "Skip the PDF parsing step (useful after running "
            "scripts/ingest_author_content.py which writes parsed text directly)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(corpus=args.corpus, profile=args.profile, skip_parse=args.skip_parse)
