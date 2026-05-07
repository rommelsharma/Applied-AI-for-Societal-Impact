"""
Baseline-vs-RAG bias detection backed by Amazon Bedrock.
"""

from __future__ import annotations

import json
from typing import Any

from app.services.bedrock_provider import BedrockProvider
from data_pipeline.concept_extractor import extract_concepts
from rag.retriever import KnowledgeRetriever
from shared_components.utilities.path_utils import get_metadata_dir, get_prompts_dir


def load_taxonomy() -> list[dict]:
    file_path = get_metadata_dir() / "bias-taxonomy.json"
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prompt() -> str:
    file_path = get_prompts_dir() / "bias_detection_system_prompt.txt"
    with file_path.open("r", encoding="utf-8") as handle:
        return handle.read()


def extract_json_payload(text: str) -> dict[str, Any]:
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
    if not retrieval_results:
        return "No supporting context was retrieved from the knowledge base."

    blocks = []
    for index, item in enumerate(retrieval_results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Context {index}]",
                    f"Source: {item.source}",
                    f"Author: {item.author}",
                    f"Chunk ID: {item.id}",
                    f"Concepts: {', '.join(item.concepts) if item.concepts else 'none'}",
                    f"Summary: {item.summary}",
                    f"Excerpt: {item.text[:900]}",
                ]
            )
        )

    return "\n\n".join(blocks)


def build_system_prompt(base_prompt: str, taxonomy: list[dict], rag_context: str | None = None) -> str:
    sections = [base_prompt]

    if rag_context:
        sections.append("Retrieved Supporting Context:\n" + rag_context)

    sections.append("Bias Taxonomy:\n" + json.dumps(taxonomy, indent=2))
    return "\n\n".join(sections)


def call_llm(provider: BedrockProvider, system_prompt: str, scenario: str) -> dict[str, Any]:
    user_prompt = f"Scenario:\n{scenario}\n\nReturn only valid JSON that follows the required schema."
    result = provider.converse(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    try:
        return extract_json_payload(result.text)
    except json.JSONDecodeError:
        # Retry once with stricter formatting instructions if the first answer is malformed.
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


def detect_bias_comparison(scenario: str) -> dict[str, Any]:
    taxonomy = load_taxonomy()
    base_prompt = load_prompt()
    provider = BedrockProvider()
    retriever = KnowledgeRetriever(provider=provider)

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
                "summary": item.summary,
            }
            for item in retrieval_results
        ],
    }


def compare_outputs(no_rag: dict[str, Any], with_rag: dict[str, Any]) -> None:
    print("\nCOMPARISON INSIGHT")
    if no_rag == with_rag:
        print("No difference detected")
    else:
        print("RAG introduced differences")

    print("Bias count (no RAG):", len(no_rag.get("biases_identified", [])))
    print("Bias count (with RAG):", len(with_rag.get("biases_identified", [])))


def print_comparison(result: dict[str, Any]) -> None:
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
