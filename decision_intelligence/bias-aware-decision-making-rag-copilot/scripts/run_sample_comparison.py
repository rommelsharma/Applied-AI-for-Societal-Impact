"""
Run two illustrative scenarios through the baseline-vs-RAG bias detector.

Produces:
    * ``evaluation/sample_results.json`` - raw payloads (without_rag, with_rag, retrieval).
    * ``docs/sample_results_comparison.md`` - reviewer-facing markdown report.

Run from the project root:
    python scripts/run_sample_comparison.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.services.bias_detector import detect_bias_comparison
from shared_components.utilities.path_utils import get_corpus_name, get_vector_store_dir


def _read_index_manifest() -> dict:
    """Return the active corpus's vector-store manifest as a dict.

    Falls back to a minimal placeholder when the manifest is missing so the
    report can still render against an unbuilt corpus during development.
    """
    manifest_path = get_vector_store_dir() / "manifest.json"
    if not manifest_path.exists():
        return {"corpus": get_corpus_name(), "count": 0, "dimension": 0, "index_type": "unknown"}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


# Two deliberately distinct scenarios to exercise different parts of the taxonomy.
SCENARIOS = [
    {
        "id": "similarity_bias_displacement",
        "title": "Similarity Bias, Favouritism, and Threat-Driven Displacement",
        "scenario": (
            "A senior manager is filling a high-visibility lead role on his team. Two internal "
            "candidates apply. Candidate A has an objectively stronger track record - higher "
            "performance ratings for three consecutive cycles, two cross-functional initiatives "
            "delivered, and consistently positive 360-degree feedback. Candidate B is less "
            "experienced and has a weaker performance history, but went to the same university "
            "as the manager, shares the manager's hobbies, and is part of the same close social "
            "circle outside work. The manager privately tells a peer that Candidate A is "
            "'too ambitious' and could become a threat to his own promotion trajectory in the "
            "next 18 months, while Candidate B is 'easier to work with' and 'more loyal'. The "
            "manager selects Candidate B and presents the decision to HR as a 'culture and team "
            "fit' call, with no scoring rubric and no documentation of how the two candidates "
            "were compared on the role criteria."
        ),
    },
    {
        "id": "legal_case_risk_score",
        "title": "Legal Case - Algorithmic Risk Score, Anchoring, and Sentencing Consistency",
        "scenario": (
            "A first-time non-violent offender is being sentenced. The pre-sentence report "
            "includes a proprietary algorithmic risk score that flags the defendant as "
            "'high risk' of re-offending, largely because the model weighs the defendant's "
            "neighbourhood, employment instability, and prior arrests (not convictions) of "
            "family members. The defence attorney notes that historical data underlying the "
            "model over-represents arrests from heavily policed neighbourhoods. The presiding "
            "judge has a reputation for handing down sentences roughly 30 percent harsher than "
            "her peers in this jurisdiction for comparable cases. The prosecution opens with a "
            "request for the maximum allowable sentence, citing a recent unrelated case in the "
            "media where a similar-sounding defendant re-offended within months. The judge "
            "reviews the risk score, the prosecution's anchor, and the media-salient comparison "
            "case, but does not consult base-rate data for first-time non-violent offenders or "
            "request a fairness review of the risk model. There is no structured sentencing "
            "rubric in use."
        ),
    },
]


def render_markdown(results: list[dict]) -> str:
    """Render the per-scenario comparison results as a reviewer-facing markdown report."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M %Z").strip()
    manifest = _read_index_manifest()
    corpus_name = manifest.get("corpus", get_corpus_name())
    chunk_count = manifest.get("count", 0)
    embedding_dim = manifest.get("dimension", 1024)
    index_type = manifest.get("index_type", "IndexFlatIP")

    lines: list[str] = []
    lines.append("# Sample Results - Baseline vs RAG Comparison")
    lines.append("")
    lines.append(
        "This document captures end-to-end outputs from the bias-aware decision-making "
        "RAG copilot for two representative scenarios. For each scenario the system was "
        "called twice with the same locked JSON-schema system prompt and the same closed "
        "bias taxonomy. The only difference between the two calls is whether retrieved "
        "context from the curated knowledge base was injected into the system prompt."
    )
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append("")
    lines.append(f"**Corpus:** `{corpus_name}` (this is the public, committed corpus; "
                 f"private full-book results are produced separately via "
                 f"`scripts/compare_versions.py` and are not included in this report).")
    lines.append("")
    lines.append("**Models:**")
    lines.append("")
    lines.append("- Chat: `us.anthropic.claude-sonnet-4-5-20250929-v1:0` (Amazon Bedrock)")
    lines.append(
        f"- Embeddings: `amazon.titan-embed-text-v2:0` ({embedding_dim} dims, normalised)"
    )
    lines.append(
        f"- Retrieval index: FAISS `{index_type}` over {chunk_count} chunks"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Table of Contents")
    lines.append("")
    for i, entry in enumerate(results, start=1):
        slug = entry["id"].replace("_", "-")
        lines.append(f"{i}. [Scenario {i} - {entry['title']}](#scenario-{i}-{slug})")
    lines.append("")
    lines.append("---")
    lines.append("")

    for index, entry in enumerate(results, start=1):
        slug = entry["id"].replace("_", "-")
        lines.append(f"## Scenario {index} - {entry['title']}")
        lines.append("")
        lines.append(f"<a id=\"scenario-{index}-{slug}\"></a>")
        lines.append("")

        lines.append("### Prompt")
        lines.append("")
        lines.append("> " + entry["scenario"].replace("\n", "\n> "))
        lines.append("")

        lines.append("### Retrieved Supporting Context (RAG Path)")
        lines.append("")
        retrieval = entry["result"].get("retrieval", [])
        if not retrieval:
            lines.append("_No retrieval results were returned for this scenario._")
        else:
            lines.append(
                "| # | Source | Author | Score | Concepts | Decision Domains |"
            )
            lines.append(
                "|---|--------|--------|-------|----------|------------------|"
            )
            for r_idx, item in enumerate(retrieval, start=1):
                concepts = ", ".join(item.get("concepts", [])) or "_none_"
                domains = ", ".join(item.get("decision_domains", [])) or "_none_"
                lines.append(
                    f"| {r_idx} | `{item.get('source', '')}` | {item.get('author', '')} | "
                    f"{item.get('score', 0):.4f} | {concepts} | {domains} |"
                )
        lines.append("")

        if retrieval:
            lines.append("**Top retrieved excerpts (summaries):**")
            lines.append("")
            for r_idx, item in enumerate(retrieval, start=1):
                lines.append(
                    f"- **[{r_idx}] {item.get('source', '')}** - {item.get('summary', '')}"
                )
            lines.append("")

        lines.append("### Baseline Output (without RAG)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(entry["result"]["without_rag"], indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

        lines.append("### RAG-Enhanced Output (with RAG)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(entry["result"]["with_rag"], indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

        baseline_biases = entry["result"]["without_rag"].get("biases_identified", [])
        rag_biases = entry["result"]["with_rag"].get("biases_identified", [])
        baseline_names = sorted({b.get("bias_name", "") for b in baseline_biases})
        rag_names = sorted({b.get("bias_name", "") for b in rag_biases})

        lines.append("### Comparison Summary")
        lines.append("")
        lines.append("| Dimension | Baseline (no RAG) | RAG-enhanced |")
        lines.append("|-----------|-------------------|--------------|")
        lines.append(f"| Biases identified (count) | {len(baseline_biases)} | {len(rag_biases)} |")
        lines.append(
            f"| Biases identified (names) | {', '.join(baseline_names) or '_none_'} | "
            f"{', '.join(rag_names) or '_none_'} |"
        )
        only_in_rag = sorted(set(rag_names) - set(baseline_names))
        only_in_baseline = sorted(set(baseline_names) - set(rag_names))
        lines.append(
            f"| Surfaced only with RAG | _n/a_ | {', '.join(only_in_rag) or '_none_'} |"
        )
        lines.append(
            f"| Surfaced only without RAG | {', '.join(only_in_baseline) or '_none_'} | _n/a_ |"
        )
        lines.append(
            f"| Recommended actions (count) | {len(entry['result']['without_rag'].get('recommended_actions', []))} | "
            f"{len(entry['result']['with_rag'].get('recommended_actions', []))} |"
        )
        lines.append(
            f"| Alternative perspectives (count) | {len(entry['result']['without_rag'].get('alternative_perspectives', []))} | "
            f"{len(entry['result']['with_rag'].get('alternative_perspectives', []))} |"
        )
        lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("## Notes on Reading These Results")
    lines.append("")
    lines.append(
        "- Both calls use temperature 0.1 for near-deterministic output."
    )
    lines.append(
        "- The system prompt is identical between baseline and RAG paths; only the retrieved "
        "context block differs."
    )
    lines.append(
        "- The model is constrained to a closed taxonomy and may not invent bias names. "
        "Differences between the two paths therefore reflect which biases the model judged "
        "salient given (or without) the retrieved context."
    )
    lines.append(
        "- The system is decision support only and is not legal, medical, or HR advice."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Run both scenarios, persist JSON, and write the markdown comparison report."""

    results: list[dict] = []

    for scenario in SCENARIOS:
        print(f"\n>>> Running scenario: {scenario['id']} - {scenario['title']}")
        result = detect_bias_comparison(scenario["scenario"])
        results.append(
            {
                "id": scenario["id"],
                "title": scenario["title"],
                "scenario": scenario["scenario"],
                "result": result,
            }
        )

        baseline_count = len(result["without_rag"].get("biases_identified", []))
        rag_count = len(result["with_rag"].get("biases_identified", []))
        retrieval_count = len(result.get("retrieval", []))
        print(
            f"    biases (no RAG): {baseline_count}, "
            f"biases (with RAG): {rag_count}, "
            f"retrieved chunks: {retrieval_count}"
        )

    json_output = PROJECT_ROOT / "evaluation" / "sample_results.json"
    json_output.parent.mkdir(parents=True, exist_ok=True)
    with json_output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved raw results to: {json_output}")

    md_output = PROJECT_ROOT / "docs" / "sample_results_comparison.md"
    md_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.write_text(render_markdown(results), encoding="utf-8")
    print(f"Saved markdown report to: {md_output}")


if __name__ == "__main__":
    main()
