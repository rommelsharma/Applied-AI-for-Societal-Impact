"""
Baseline-vs-RAG bias detection orchestration.

This module is the runtime brain of the copilot. For every user scenario it:
    1. Extracts taxonomy concepts from the scenario text.
    2. Retrieves supporting context from the curated knowledge base
       (corpus-aware: defaults to ``public``; can be pinned to ``private`` for
       internal evaluation against the full-book corpus).
    3. Calls the chat model twice - once without retrieval (baseline) and once
       with retrieval (RAG-enhanced) - using the same locked JSON-schema prompt.
    4. Returns both answers plus a traceability record of what was retrieved.

Running both prompts on every scenario is intentional: it lets the project
demonstrate, measurably, the lift that RAG adds over a vanilla LLM call.
"""

from __future__ import annotations

import json
from typing import Any

from app.services.bedrock_provider import BedrockProvider
from data_pipeline.concept_extractor import extract_concepts
from rag.retriever import KnowledgeRetriever
from shared_components.utilities.path_utils import get_metadata_dir, get_prompts_dir


def load_taxonomy() -> list[dict]:
    """Load the bias taxonomy JSON used to constrain the LLM output."""
    file_path = get_metadata_dir() / "bias-taxonomy.json"
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prompt() -> str:
    """Load the system-prompt template that defines role, rules, and output schema."""
    file_path = get_prompts_dir() / "bias_detection_system_prompt.txt"
    with file_path.open("r", encoding="utf-8") as handle:
        return handle.read()


def extract_json_payload(text: str) -> dict[str, Any]:
    """Tolerantly parse the model's response into a JSON object.

    Strips a leading ```json fence if present, and falls back to slicing
    from the first ``{`` to the last ``}`` if the response has trailing prose.
    """
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


def format_retrieved_context(retrieval_results) -> str:
    """Render retrieval results as numbered context blocks for the system prompt.

    Includes the upgraded metadata (chapter_title, passage_type,
    decision_phase) so the LLM can prefer prescriptive content when the user
    asks "what should I do?" and so any audit reader can see exactly what kind
    of passage shaped the answer.
    """
    if not retrieval_results:
        return "No supporting context was retrieved from the knowledge base."

    blocks = []
    for index, item in enumerate(retrieval_results, start=1):
        location = f"{item.source}"
        if getattr(item, "chapter_title", ""):
            location += f" / {item.chapter_title}"
        blocks.append(
            "\n".join(
                [
                    f"[Context {index}]",
                    f"Source: {location}",
                    f"Author: {item.author}",
                    f"Chunk ID: {item.id}",
                    f"Passage type: {getattr(item, 'passage_type', 'unknown')} | "
                    f"Decision phase: {getattr(item, 'decision_phase', 'unknown')}",
                    f"Concepts: {', '.join(item.concepts) if item.concepts else 'none'}",
                    f"Summary: {item.summary}",
                    f"Excerpt: {item.text[:900]}",
                ]
            )
        )
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

    return "\n\n".join(blocks)


def build_system_prompt(base_prompt: str, taxonomy: list[dict], rag_context: str | None = None) -> str:
    """Assemble the final system prompt sent to the LLM.

    Layout:
        * the locked role/schema/rules text
        * (optional) retrieved context block - only included for the RAG path
        * the bias taxonomy (closed vocabulary the model must use)
    """
    sections = [base_prompt]

    if rag_context:
        sections.append("Retrieved Supporting Context:\n" + rag_context)

    sections.append("Bias Taxonomy:\n" + json.dumps(taxonomy, indent=2))
    return "\n\n".join(sections)


def call_llm(provider: BedrockProvider, system_prompt: str, scenario: str) -> dict[str, Any]:
    """Invoke the LLM with a single scenario and return parsed JSON.

    A short, stricter retry is performed if the first response is not valid
    JSON. The retry tightens the formatting instructions and bumps the token
    cap so a near-miss (e.g. a missing closing brace at the limit) gets a
    second chance rather than failing the whole comparison run.
    """
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
    """End-to-end orchestration for one scenario.

    Args:
        scenario: The user-supplied scenario text.
        corpus: Optional corpus override. ``None`` defers to ``CORPUS_NAME``
            env var (default: ``public``). Pass ``"private"`` to evaluate
            against the local-only full-book corpus.

    Returns:
        Dict with three keys: ``without_rag``, ``with_rag``, ``retrieval``.
    """
    taxonomy = load_taxonomy()
    base_prompt = load_prompt()
    provider = BedrockProvider()
    retriever = KnowledgeRetriever(provider=provider, corpus=corpus)

    scenario_concepts = extract_concepts(scenario)
    retrieval_results = retriever.search(
        scenario,
        preferred_concepts=scenario_concepts["concepts"],
    )
    rag_context = format_retrieved_context(retrieval_results)

    system_prompt_no_rag = build_system_prompt(base_prompt, taxonomy)
    response_no_rag = call_llm(provider, system_prompt_no_rag, scenario)

    system_prompt_with_rag = build_system_prompt(base_prompt, taxonomy, rag_context)
    response_with_rag = call_llm(provider, system_prompt_with_rag, scenario)

    return {
        "corpus": retriever.corpus,
        "without_rag": response_no_rag,
        "with_rag": response_with_rag,
        "retrieval": [
            {
                "id": item.id,
                "source": item.source,
                "author": item.author,
                "score": item.score,
                "concepts": item.concepts,
                "decision_domains": item.decision_domains,
                "chapter_title": getattr(item, "chapter_title", ""),
                "passage_type": getattr(item, "passage_type", "unknown"),
                "decision_phase": getattr(item, "decision_phase", "unknown"),
                "summary": item.summary,
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
