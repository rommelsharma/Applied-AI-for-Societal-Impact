"""
Taxonomy-driven concept extraction for knowledge chunks and user scenarios.
"""

from __future__ import annotations

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

INPUT_FILE = get_chunks_dir() / "chunks.json"
OUTPUT_DIR = ensure_directory(get_chunks_dir())
OUTPUT_FILE = OUTPUT_DIR / "chunks_with_concepts.json"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_concepts(text: str) -> dict:
    normalized_text = normalize_text(text)
    catalog = build_concept_catalog()

    detected_concepts = []
    confidence_scores = {}

    for concept_name, entry in catalog.items():
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


def process_chunks():
    print(f"\nReading chunks from:\n{INPUT_FILE}\n")

    with INPUT_FILE.open("r", encoding="utf-8") as handle:
        chunks = json.load(handle)

    processed_chunks = []

    for chunk in chunks:
        extraction_result = extract_concepts(chunk["text"])
        chunk["concepts"] = extraction_result["concepts"]
        chunk["concept_confidence"] = extraction_result["confidence"]
        processed_chunks.append(chunk)

    print(f"Processed chunks: {len(processed_chunks)}")
    print(f"\nSaving enriched chunks to:\n{OUTPUT_FILE}")

    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(processed_chunks, handle, indent=2)

    print("\n✅ Concept extraction complete")


if __name__ == "__main__":
    process_chunks()
