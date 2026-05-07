"""
Master orchestration pipeline for ingestion and retrieval preparation.
"""

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


def run_pipeline():
    print("\n==============================")
    print("STEP 1 — PARSING PDFs")
    print("==============================")
    parse_all_pdfs()

    print("\n==============================")
    print("STEP 2 — CREATING CHUNKS")
    print("==============================")
    create_chunks()

    print("\n==============================")
    print("STEP 3 — EXTRACTING CONCEPTS")
    print("==============================")
    process_chunks()

    print("\n==============================")
    print("STEP 4 — ENRICHING KNOWLEDGE")
    print("==============================")
    enrich()

    print("\n✅ KNOWLEDGE BASE PIPELINE COMPLETE")


if __name__ == "__main__":
    run_pipeline()
