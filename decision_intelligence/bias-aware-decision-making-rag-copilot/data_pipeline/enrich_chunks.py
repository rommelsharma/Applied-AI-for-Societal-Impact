"""
Knowledge-base enricher - step 4 of the offline ingestion pipeline.

Promotes concept-tagged chunks into the final retrieval-ready knowledge base
written to ``<corpus>/knowledge/knowledge_base.json``.

For every chunk this module derives:
    * ``summary``                     - short truncated text used in retrieved-context blocks
    * ``importance``                  - high/medium/low aggregated from concept metadata
    * ``decision_domains``            - union of the decision domains tied to each concept
    * ``keywords``                    - top frequency words excluding common stopwords
    * ``passage_type`` /
      ``decision_phase``              - lightweight passage classification (rule-based by
                                        default, LLM-upgradable via ``ENRICHMENT_USE_LLM``)
    * ``chapter_title``               - inherited from the chunker when a chapter
                                        boundary was detected
    * ``chunking_profile``            - inherited from the chunker so retrieval lift
                                        analysis can group results by profile

Together with the traceability fields inherited from the chunker (id, source,
author, title, year, parser_notes, cleaning_notes), this is what the embedder
ultimately reads in the next step.
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

from data_pipeline.passage_classifier import classify_passage
from shared_components.utilities.path_utils import (
    ensure_directory,
    get_chunks_dir,
    get_knowledge_dir,
)
from shared_components.utilities.taxonomy_utils import build_concept_catalog


def clean_summary_text(text: str) -> str:
    """Collapse whitespace so summaries display predictably in the system prompt."""
    return re.sub(r"\s+", " ", text).strip()


def generate_summary(text: str, max_length: int = 280) -> str:
    """Produce a short summary of a chunk by truncating to ``max_length`` chars.

    A trailing ellipsis is added when truncation occurs so the summary clearly
    indicates that the underlying chunk is longer.
    """
    cleaned = clean_summary_text(text)
    return cleaned if len(cleaned) <= max_length else cleaned[:max_length].rstrip() + "..."


def determine_importance(concepts: list[str]) -> str:
    """Roll up per-concept importance into a chunk-level signal.

    A chunk inherits the highest importance level among its concepts. Chunks
    with three or more medium-importance concepts are upgraded to ``medium``
    on the assumption that breadth implies retrieval value.
    """
    catalog = build_concept_catalog()
    levels = {catalog.get(concept, {}).get("importance", "low") for concept in concepts}

    if "high" in levels:
        return "high"
    if "medium" in levels or len(concepts) >= 3:
        return "medium"
    return "low"


def derive_decision_domains(concepts: list[str]) -> list[str]:
    """Compute the set of decision domains that a chunk applies to.

    Each concept in the catalog declares which decision domains it is
    relevant to (hr, finance, legal, governance, ai, ...). The chunk inherits
    the union of those domains, sorted for deterministic output.
    """
    catalog = build_concept_catalog()
    domains = set()

    for concept in concepts:
        for domain in catalog.get(concept, {}).get("decision_domains", []):
            domains.add(domain)

    return sorted(domains)


def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """Return the top ``max_keywords`` content words from a chunk by frequency.

    A small stopword list filters obvious noise. This is a lightweight signal
    used for inspection and future heuristic filters; the heavy lifting still
    happens via embeddings and concept tags.
    """
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


def enrich(corpus: str | None = None):
    """Apply all enrichment steps and write the final knowledge base file.

    Each output row is a self-contained, retrieval-ready record. The embedder
    in the next step iterates this file directly.
    """
    chunks_dir = get_chunks_dir(corpus)
    output_dir = ensure_directory(get_knowledge_dir(corpus))
    input_file = chunks_dir / "chunks_with_concepts.json"
    output_file = output_dir / "knowledge_base.json"

    print(f"\nReading chunks from:\n{input_file}\n")

    with input_file.open("r", encoding="utf-8") as handle:
        chunks = json.load(handle)

    enriched_chunks = []

    for chunk in chunks:
        concepts = chunk.get("concepts", [])
        passage_tags = classify_passage(chunk["text"])
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
                "chapter_title": chunk.get("chapter_title", ""),
                "chunking_profile": chunk.get("chunking_profile", "dossier"),
                "text": chunk["text"],
                "summary": generate_summary(chunk["text"]),
                "concepts": concepts,
                "concept_confidence": chunk.get("concept_confidence", {}),
                "importance": determine_importance(concepts),
                "decision_domains": derive_decision_domains(concepts),
                "keywords": extract_keywords(chunk["text"]),
                "passage_type": passage_tags["passage_type"],
                "passage_confidence": passage_tags["passage_confidence"],
                "decision_phase": passage_tags["decision_phase"],
                "decision_phase_confidence": passage_tags["decision_phase_confidence"],
                "classifier": passage_tags["classifier"],
                "word_count": chunk.get("word_count", len(chunk["text"].split())),
                "chunk_index": chunk.get("chunk_index", -1),
            }
        )

    print(f"Enriched chunks: {len(enriched_chunks)}")
    print(f"\nSaving knowledge base to:\n{output_file}")

    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(enriched_chunks, handle, indent=2)

    print("\n✅ Knowledge base created successfully")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. ``--corpus`` overrides the env-driven default."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=None, help="Corpus to enrich (defaults to CORPUS_NAME or 'public').")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    enrich(corpus=args.corpus)
