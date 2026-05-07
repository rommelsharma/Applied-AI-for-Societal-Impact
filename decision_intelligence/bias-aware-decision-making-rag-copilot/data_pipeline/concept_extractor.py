"""
Concept extractor - step 3 of the offline ingestion pipeline.

Tags every chunk with the taxonomy concepts whose canonical name or aliases
appear in the chunk's text. The same function is also called at runtime on
the user's scenario so the retriever can prefer chunks that share concepts
with the query.

Why a taxonomy-driven layer on top of pure semantic search:
    * It gives every chunk a structured, symbolic handle ("anchoring_bias",
      "decision_hygiene") in addition to its dense vector.
    * It lets the retriever filter by concept or decision domain when those
      filters are useful, and fall back to pure semantic search when not.
    * It keeps the LLM constrained to a closed vocabulary at generation time.
"""

from __future__ import annotations

import argparse
import json
import re

from pathlib import Path

try:
    from data_pipeline.bootstrap import add_project_root_to_sys_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_pipeline.bootstrap import add_project_root_to_sys_path

add_project_root_to_sys_path()

from shared_components.utilities.path_utils import ensure_directory, get_chunks_dir
from shared_components.utilities.taxonomy_utils import build_concept_catalog, normalize_phrase


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace so alias matching is consistent."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_concepts(text: str) -> dict:
    """Identify which taxonomy concepts appear in a piece of text.

    For each concept the catalog provides the canonical name plus a list of
    aliases. A concept is recorded when *any* of its aliases is found in the
    normalised text. Confidence is a coarse signal: ``high`` if two or more
    aliases match (multiple framings agree), ``medium`` for a single hit.

    Returns a dict ``{"concepts": [...sorted unique names...], "confidence": {...}}``
    consumed both at indexing time (chunks) and at runtime (user scenarios).
    """
    normalized_text = normalize_text(text)
    catalog = build_concept_catalog()

    detected_concepts = []
    confidence_scores = {}

    for concept_name, entry in catalog.items():
        # Build a deduplicated alias set including the canonical name itself,
        # so a chunk that uses only the canonical phrase still matches.
        aliases = {normalize_phrase(concept_name), *[normalize_phrase(alias) for alias in entry["aliases"]]}
        matches = sum(1 for alias in aliases if alias in normalized_text)

        if matches == 0:
            continue

        detected_concepts.append(concept_name)
        confidence_scores[concept_name] = "high" if matches >= 2 else "medium"

    return {
        "concepts": sorted(set(detected_concepts)),
        "confidence": confidence_scores,
    }


def process_chunks(corpus: str | None = None):
    """Apply :func:`extract_concepts` to every chunk and write the enriched JSON."""
    chunks_dir = ensure_directory(get_chunks_dir(corpus))
    input_file = chunks_dir / "chunks.json"
    output_file = chunks_dir / "chunks_with_concepts.json"

    print(f"\nReading chunks from:\n{input_file}\n")

    with input_file.open("r", encoding="utf-8") as handle:
        chunks = json.load(handle)

    processed_chunks = []

    for chunk in chunks:
        extraction_result = extract_concepts(chunk["text"])
        chunk["concepts"] = extraction_result["concepts"]
        chunk["concept_confidence"] = extraction_result["confidence"]
        processed_chunks.append(chunk)

    print(f"Processed chunks: {len(processed_chunks)}")
    print(f"\nSaving enriched chunks to:\n{output_file}")

    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(processed_chunks, handle, indent=2)

    print("\n✅ Concept extraction complete")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. ``--corpus`` overrides the env-driven default."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=None, help="Corpus to process (defaults to CORPUS_NAME or 'public').")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_chunks(corpus=args.corpus)
