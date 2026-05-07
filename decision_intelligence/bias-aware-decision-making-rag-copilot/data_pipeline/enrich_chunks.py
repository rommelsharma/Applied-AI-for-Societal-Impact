"""
Final enrichment of chunks before embedding and retrieval.
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

from shared_components.utilities.path_utils import (
    ensure_directory,
    get_chunks_dir,
    get_knowledge_dir,
)
from shared_components.utilities.taxonomy_utils import build_concept_catalog

INPUT_FILE = get_chunks_dir() / "chunks_with_concepts.json"
OUTPUT_DIR = ensure_directory(get_knowledge_dir())
OUTPUT_FILE = OUTPUT_DIR / "knowledge_base.json"


def clean_summary_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def generate_summary(text: str, max_length: int = 280) -> str:
    cleaned = clean_summary_text(text)
    return cleaned if len(cleaned) <= max_length else cleaned[:max_length].rstrip() + "..."


def determine_importance(concepts: list[str]) -> str:
    catalog = build_concept_catalog()
    levels = {catalog.get(concept, {}).get("importance", "low") for concept in concepts}

    if "high" in levels:
        return "high"
    if "medium" in levels or len(concepts) >= 3:
        return "medium"
    return "low"


def derive_decision_domains(concepts: list[str]) -> list[str]:
    catalog = build_concept_catalog()
    domains = set()

    for concept in concepts:
        for domain in catalog.get(concept, {}).get("decision_domains", []):
            domains.add(domain)

    return sorted(domains)


def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    stopwords = {
        "about", "after", "also", "among", "because", "being", "from", "into",
        "much", "only", "than", "that", "their", "there", "these", "they",
        "this", "those", "when", "where", "which", "with", "would", "your",
        "the", "and", "of", "to", "a", "in", "is", "for", "on", "as", "by", "an", "are", "at",
    }

    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    frequency: dict[str, int] = {}

    for word in words:
        if word in stopwords:
            continue
        frequency[word] = frequency.get(word, 0) + 1

    sorted_words = sorted(frequency.items(), key=lambda item: item[1], reverse=True)
    return [word for word, _ in sorted_words[:max_keywords]]


def enrich():
    print(f"\nReading chunks from:\n{INPUT_FILE}\n")

    with INPUT_FILE.open("r", encoding="utf-8") as handle:
        chunks = json.load(handle)

    enriched_chunks = []

    for chunk in chunks:
        concepts = chunk.get("concepts", [])
        enriched_chunks.append(
            {
                "id": chunk["id"],
                "source": chunk["source"],
                "file_name": chunk.get("file_name"),
                "author": chunk.get("author", "unknown"),
                "title": chunk.get("title"),
                "publication_year": chunk.get("publication_year"),
                "parser_notes": chunk.get("parser_notes", []),
                "cleaning_notes": chunk.get("cleaning_notes", []),
                "text": chunk["text"],
                "summary": generate_summary(chunk["text"]),
                "concepts": concepts,
                "concept_confidence": chunk.get("concept_confidence", {}),
                "importance": determine_importance(concepts),
                "decision_domains": derive_decision_domains(concepts),
                "keywords": extract_keywords(chunk["text"]),
                "word_count": chunk.get("word_count", len(chunk["text"].split())),
                "chunk_index": chunk.get("chunk_index", -1),
            }
        )

    print(f"Enriched chunks: {len(enriched_chunks)}")
    print(f"\nSaving knowledge base to:\n{OUTPUT_FILE}")

    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(enriched_chunks, handle, indent=2)

    print("\n✅ Knowledge base created successfully")


if __name__ == "__main__":
    enrich()
