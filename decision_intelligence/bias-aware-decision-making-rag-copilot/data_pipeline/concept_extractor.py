"""
Concept extractor - step 3 of the offline ingestion pipeline.

Tags every chunk with ontology concepts (aliases + canonical names), expands
the decision-intelligence graph into symbolic layers, and emits structured
fields consumed by enrichment and hybrid retrieval.
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
from shared_components.utilities.taxonomy_utils import (
    build_concept_catalog,
    load_decision_intelligence_ontology,
    normalize_phrase,
)


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace so alias matching is consistent."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _symbolic_from_matches(matched: list[str], catalog: dict) -> dict:
    """Aggregate 8-layer ontology fields + graph expansion for matched concept ids."""
    onto = load_decision_intelligence_ontology()
    cmap = onto.get("concepts") or {}
    if not isinstance(cmap, dict):
        cmap = {}

    biases: set[str] = set()
    cognitive_mechanisms: set[str] = set()
    failure_modes: set[str] = set()
    group_dynamics: set[str] = set()
    decision_phases: set[str] = set()
    interventions: set[str] = set()
    evidence_types: set[str] = set()
    related: set[str] = set()
    mitigated_by: set[str] = set()

    for concept_id in matched:
        node = cmap.get(concept_id) if isinstance(concept_id, str) else None
        entry = catalog.get(concept_id, {})
        if not isinstance(node, dict) or not node:
            kind = entry.get("kind", "")
            if kind in ("bias", "cognitive_mechanism"):
                biases.add(concept_id)
                cognitive_mechanisms.add(concept_id)
            elif kind in ("support_concept", "framework", "intervention"):
                interventions.add(concept_id)
            continue

        kind = node.get("kind") or entry.get("kind", "")

        if kind in ("cognitive_mechanism", "bias"):
            biases.add(concept_id)
            cognitive_mechanisms.add(concept_id)
        if isinstance(node, dict):
            for r in node.get("related_concepts") or []:
                if isinstance(r, str):
                    related.add(r)
            for m in node.get("mitigated_by") or []:
                if isinstance(m, str):
                    mitigated_by.add(m)
            for f in node.get("failure_modes") or []:
                if isinstance(f, str):
                    failure_modes.add(f)
            for g in node.get("group_dynamics") or []:
                if isinstance(g, str):
                    group_dynamics.add(g)
            for dp in node.get("decision_phases") or []:
                if isinstance(dp, str):
                    decision_phases.add(dp)
            for iv in node.get("interventions") or []:
                if isinstance(iv, str):
                    interventions.add(iv)
            for ev in node.get("evidence_type") or []:
                if isinstance(ev, str):
                    evidence_types.add(ev)

    # Support / framework concepts often act as interventions when matched directly
    for concept_id in matched:
        entry = catalog.get(concept_id, {})
        if entry.get("kind") in ("framework", "intervention", "support_concept"):
            interventions.add(concept_id)

    confidence = "high" if len(matched) >= 3 else ("medium" if matched else "low")

    return {
        "biases": sorted(biases),
        "cognitive_mechanisms": sorted(cognitive_mechanisms),
        "failure_modes": sorted(failure_modes),
        "group_dynamics": sorted(group_dynamics),
        "decision_phases": sorted(decision_phases),
        "interventions": sorted(interventions),
        "evidence_type": sorted(evidence_types)[0] if evidence_types else "",
        "confidence_level": confidence,
        "related_concepts": sorted(related),
        "mitigated_by_concepts": sorted(mitigated_by),
    }


def extract_concepts(text: str) -> dict:
    """Identify taxonomy concepts and symbolic ontology layers in ``text``."""
    normalized_text = normalize_text(text)
    catalog = build_concept_catalog()

    detected_concepts: list[str] = []
    confidence_scores: dict[str, str] = {}

    for concept_name, entry in catalog.items():
        aliases = {normalize_phrase(concept_name), *[normalize_phrase(alias) for alias in entry["aliases"]]}
        matches = sum(1 for alias in aliases if alias in normalized_text)

        if matches == 0:
            continue

        detected_concepts.append(concept_name)
        confidence_scores[concept_name] = "high" if matches >= 2 else "medium"

    matched = sorted(set(detected_concepts))
    symbolic = _symbolic_from_matches(matched, catalog)

    return {
        "concepts": matched,
        "confidence": confidence_scores,
        **symbolic,
        "ontology_query_expansion": sorted(set(matched) | set(symbolic["related_concepts"]) | set(symbolic["mitigated_by_concepts"])),
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
        extraction = extract_concepts(chunk["text"])
        chunk["concepts"] = extraction["concepts"]
        chunk["concept_confidence"] = extraction["confidence"]
        chunk["biases"] = extraction["biases"]
        chunk["cognitive_mechanisms"] = extraction["cognitive_mechanisms"]
        chunk["failure_modes"] = extraction["failure_modes"]
        chunk["group_dynamics"] = extraction["group_dynamics"]
        chunk["ontology_decision_phases"] = extraction["decision_phases"]
        chunk["interventions"] = extraction["interventions"]
        chunk["ontology_evidence_type"] = extraction["evidence_type"]
        chunk["ontology_confidence_level"] = extraction["confidence_level"]
        chunk["related_concepts"] = extraction["related_concepts"]
        chunk["mitigated_by_concepts"] = extraction["mitigated_by_concepts"]
        chunk["ontology_query_expansion"] = extraction["ontology_query_expansion"]
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
