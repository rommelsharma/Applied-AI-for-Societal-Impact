"""
Baseline-vs-RAG bias detection orchestration.

Runs taxonomy-grounded retrieval (hybrid dense + BM25 when configured),
optional synthesis blocks, and dual Claude calls for measurable lift.
"""

from __future__ import annotations

import json
from typing import Any

from app.services.bedrock_provider import BedrockProvider
from data_pipeline.concept_extractor import extract_concepts
from rag.hierarchical_retriever import (
    format_synthesis_blocks,
    load_synthesis,
    select_synthesis_blocks,
)
from rag.query_classifier import classify_query_intent
from rag.retriever import KnowledgeRetriever
from shared_components.settings import RAG_SETTINGS
from shared_components.utilities.path_utils import (
    get_bias_detection_system_prompt_path,
    get_metadata_dir,
    resolve_synthesis_json,
)


def load_taxonomy() -> list[dict]:
    """Load the bias taxonomy JSON used to constrain the LLM output."""
    file_path = get_metadata_dir() / "bias-taxonomy.json"
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prompt() -> str:
    """Load the system-prompt template that defines role, rules, and output schema."""
    file_path = get_bias_detection_system_prompt_path()
    with file_path.open("r", encoding="utf-8") as handle:
        return handle.read()


def extract_json_payload(text: str) -> dict[str, Any]:
    """Tolerantly parse the model's response into a JSON object."""
    stripped = text.strip()

    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json\n", "", 1).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(stripped[start : end + 1])


def format_retrieved_context(retrieval_results, synthesis_text: str = "") -> str:
    """Render retrieval results + optional synthesis as numbered context blocks."""
    parts: list[str] = []
    if synthesis_text.strip():
        parts.append("Corpus synthesis (high-level, evidence hierarchy aware):\n" + synthesis_text.strip())

    if not retrieval_results:
        if parts:
            return "\n\n".join(parts) + "\n\nNo supporting chunk context was retrieved."
        return "No supporting context was retrieved from the knowledge base."

    blocks = []
    for index, item in enumerate(retrieval_results, start=1):
        location = f"{item.source}"
        if getattr(item, "chapter_title", ""):
            location += f" / {item.chapter_title}"
        onto_bits = []
        if getattr(item, "biases", None):
            onto_bits.append(f"biases={item.biases}")
        if getattr(item, "failure_modes", None):
            onto_bits.append(f"failure_modes={item.failure_modes}")
        if getattr(item, "interventions", None):
            onto_bits.append(f"interventions={item.interventions}")
        onto_line = " | ".join(onto_bits) if onto_bits else ""

        base_lines = [
                    f"[Context {index}]",
                    f"Source: {location}",
                    f"Author: {item.author}",
                    f"Chunk ID: {item.id}",
                    f"Retrieval: {getattr(item, 'retrieval_method', 'dense')} | "
                    f"dense_score≈{item.score:.4f} | bm25={getattr(item, 'bm25_score', 0):.4f} | rrf={getattr(item, 'rrf_score', 0):.4f}",
                    f"Passage type: {getattr(item, 'passage_type', 'unknown')} | "
                    f"Decision phase: {getattr(item, 'decision_phase', 'unknown')}",
                    f"Concepts: {', '.join(item.concepts) if item.concepts else 'none'}",
                ]
        if onto_line:
            base_lines.append(onto_line)
        base_lines.extend(
            [
                f"Summary: {item.summary}",
                f"Excerpt: {item.text[:900]}",
            ]
        )
        blocks.append("\n".join(base_lines))
        neighbors = getattr(item, "neighbor_blocks", None) or []
        if neighbors:
            nb_lines = [
                "Adjacent context (same book, supporting continuation only — cite the primary Chunk ID above):"
            ]
            for nb in neighbors:
                nb_lines.append(
                    f"  — Neighbour chunk {nb.get('chunk_id', '')} (Δindex {nb.get('chunk_index_delta', '')}): "
                    f"{nb.get('excerpt', '')}"
                )
            blocks.append("\n".join(nb_lines))

    chunk_blob = "\n\n".join(blocks)
    if parts:
        return "\n\n".join(parts) + "\n\n" + chunk_blob
    return chunk_blob


def build_system_prompt(base_prompt: str, taxonomy: list[dict], rag_context: str | None = None) -> str:
    """Assemble the final system prompt sent to the LLM."""
    sections = [base_prompt]

    if rag_context:
        sections.append("Retrieved Supporting Context:\n" + rag_context)

    sections.append("Bias Taxonomy:\n" + json.dumps(taxonomy, indent=2))
    return "\n\n".join(sections)


def call_llm(provider: BedrockProvider, system_prompt: str, scenario: str) -> dict[str, Any]:
    """Invoke the LLM with a single scenario and return parsed JSON."""
    user_prompt = f"Scenario:\n{scenario}\n\nReturn only valid JSON that follows the required schema."
    result = provider.converse(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    try:
        return extract_json_payload(result.text)
    except json.JSONDecodeError:
        retry_prompt = (
            f"{user_prompt}\n\n"
            "Your previous answer was not valid JSON. "
            "Retry with a compact response that is strictly valid JSON, with properly closed quotes, "
            "commas, brackets, and braces."
        )
        retry_result = provider.converse(
            system_prompt=system_prompt,
            user_prompt=retry_prompt,
            max_tokens=2200,
        )
        return extract_json_payload(retry_result.text)


def detect_bias_comparison(scenario: str, *, corpus: str | None = None) -> dict[str, Any]:
    """End-to-end orchestration for one scenario."""
    taxonomy = load_taxonomy()
    base_prompt = load_prompt()
    provider = BedrockProvider()
    retriever = KnowledgeRetriever(provider=provider, corpus=corpus)

    scenario_info = extract_concepts(scenario)
    concept_keys = scenario_info.get("ontology_query_expansion") or scenario_info["concepts"]

    qc: dict[str, Any] = {"intent": "recall", "mmr_lambda": RAG_SETTINGS.mmr_lambda}
    if RAG_SETTINGS.query_classification_enabled:
        qc = classify_query_intent(scenario)

    synthesis_text = ""
    if RAG_SETTINGS.synthesis_context_enabled:
        syn_path = resolve_synthesis_json(retriever.corpus)
        syn_doc = load_synthesis(syn_path)
        blocks = select_synthesis_blocks(syn_doc, scenario_info.get("concepts") or [])
        synthesis_text = format_synthesis_blocks(blocks)

    retrieval_results = retriever.search(
        scenario,
        preferred_concepts=concept_keys,
        mmr_lambda=float(qc.get("mmr_lambda") or RAG_SETTINGS.mmr_lambda),
    )
    rag_context = format_retrieved_context(retrieval_results, synthesis_text=synthesis_text)

    system_prompt_no_rag = build_system_prompt(base_prompt, taxonomy)
    response_no_rag = call_llm(provider, system_prompt_no_rag, scenario)

    system_prompt_with_rag = build_system_prompt(base_prompt, taxonomy, rag_context)
    response_with_rag = call_llm(provider, system_prompt_with_rag, scenario)

    return {
        "corpus": retriever.corpus,
        "without_rag": response_no_rag,
        "with_rag": response_with_rag,
        "query_classification": qc,
        "retrieval": [
            {
                "id": item.id,
                "source": item.source,
                "author": item.author,
                "score": item.score,
                "bm25_score": getattr(item, "bm25_score", 0.0),
                "rrf_score": getattr(item, "rrf_score", 0.0),
                "retrieval_method": getattr(item, "retrieval_method", "dense"),
                "concepts": item.concepts,
                "biases": getattr(item, "biases", []),
                "failure_modes": getattr(item, "failure_modes", []),
                "interventions": getattr(item, "interventions", []),
                "decision_domains": item.decision_domains,
                "chapter_title": getattr(item, "chapter_title", ""),
                "passage_type": getattr(item, "passage_type", "unknown"),
                "decision_phase": getattr(item, "decision_phase", "unknown"),
                "summary": item.summary,
                "text_excerpt": (item.text or "")[:600],
                "neighbor_blocks": list(getattr(item, "neighbor_blocks", None) or []),
            }
            for item in retrieval_results
        ],
    }


def compare_outputs(no_rag: dict[str, Any], with_rag: dict[str, Any]) -> None:
    """Print a lightweight diff summary between the two model outputs."""
    print("\nCOMPARISON INSIGHT")
    if no_rag == with_rag:
        print("No difference detected")
    else:
        print("RAG introduced differences")

    print("Bias count (no RAG):", len(no_rag.get("biases_identified", [])))
    print("Bias count (with RAG):", len(with_rag.get("biases_identified", [])))


def print_comparison(result: dict[str, Any]) -> None:
    """Pretty-print the full comparison payload for interactive runs."""
    print(f"\nCorpus: {result.get('corpus', 'unknown')}")
    print("\n" + "=" * 80)
    print("WITHOUT RAG (Baseline)")
    print("=" * 80)
    print(json.dumps(result["without_rag"], indent=2))

    print("\n" + "=" * 80)
    print("WITH RAG (Enhanced)")
    print("=" * 80)
    print(json.dumps(result["with_rag"], indent=2))

    print("\n" + "=" * 80)
    print("RETRIEVED CONTEXT")
    print("=" * 80)
    print(json.dumps(result["retrieval"], indent=2))
