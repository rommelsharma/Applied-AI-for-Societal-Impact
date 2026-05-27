"""
Side-by-side corpus version comparison.

Runs the full evaluation suite (or any subset) against two named corpora and
emits a markdown summary that quantifies the lift each corpus version
delivers over the baseline (no-RAG) call.

Typical use cases:

    # Public dossiers vs private full-book corpus, on every scenario.
    python scripts/compare_versions.py --versions public,private

    # Same, but limited to two scenario ids for a quick smoke test.
    python scripts/compare_versions.py --versions public,private --scenario-ids 11,12

The script *never* reveals private-corpus content in the public report -
the markdown summary aggregates by counts and named biases only. Use the
JSON sidecar (``evaluation/version_comparison_<timestamp>.json``) for full
forensic detail when needed.

Output layout:

    evaluation/
      version_comparison_<timestamp>.json        # raw per-scenario payloads
      version_comparison_<timestamp>.md          # reviewer-friendly markdown summary

This script complements ``run_sample_comparison.py`` (one corpus, two
scenarios) and ``run_evaluation.py`` (one corpus, all scenarios) - it is the
one to use when measuring index lift across corpus versions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.services.bias_detector import detect_bias_comparison
from shared_components.utilities.path_utils import get_corpus_dir, get_evaluation_dir, resolve_knowledge_base_json


def _load_scenarios(scenario_ids: list[int] | None) -> list[dict]:
    """Load the canonical extended suite and optionally filter by id."""
    from evaluation.scenario_catalog import get_extended_test_scenarios

    scenarios = get_extended_test_scenarios()
    if scenario_ids:
        wanted = set(scenario_ids)
        scenarios = [s for s in scenarios if s["id"] in wanted]
    return scenarios


def _corpus_exists(corpus: str) -> bool:
    """True iff the named corpus has a knowledge base on disk (v4 or legacy path)."""
    return resolve_knowledge_base_json(corpus).exists()


def _bias_names(payload: dict) -> set[str]:
    """Return the set of bias names surfaced in a single LLM response."""
    return {entry.get("bias_name", "") for entry in payload.get("biases_identified", [])}


def _summarise_run(scenario: dict, result: dict) -> dict:
    """Compress one scenario's full payload into a compact comparison record."""
    baseline_names = _bias_names(result["without_rag"])
    rag_names = _bias_names(result["with_rag"])
    return {
        "id": scenario["id"],
        "domain": scenario.get("domain", ""),
        "tags": scenario.get("tags", []),
        "biases_baseline": sorted(baseline_names),
        "biases_rag": sorted(rag_names),
        "added_by_rag": sorted(rag_names - baseline_names),
        "removed_by_rag": sorted(baseline_names - rag_names),
        "recommended_actions_baseline": len(result["without_rag"].get("recommended_actions", [])),
        "recommended_actions_rag": len(result["with_rag"].get("recommended_actions", [])),
        "retrieved_count": len(result.get("retrieval", [])),
        "top_retrieval_concepts": list(
            Counter(
                concept
                for chunk in result.get("retrieval", [])
                for concept in chunk.get("concepts", [])
            ).most_common(5)
        ),
    }


def _aggregate(records: list[dict]) -> dict:
    """Compute simple aggregate metrics over a list of per-scenario summaries."""
    if not records:
        return {}
    return {
        "scenario_count": len(records),
        "avg_biases_baseline": round(mean(len(r["biases_baseline"]) for r in records), 2),
        "avg_biases_rag": round(mean(len(r["biases_rag"]) for r in records), 2),
        "avg_actions_baseline": round(mean(r["recommended_actions_baseline"] for r in records), 2),
        "avg_actions_rag": round(mean(r["recommended_actions_rag"] for r in records), 2),
        "scenarios_where_rag_added_biases": sum(1 for r in records if r["added_by_rag"]),
        "avg_retrieval_count": round(mean(r["retrieved_count"] for r in records), 2),
    }


def _render_markdown(per_version: dict, scenarios: list[dict], timestamp: str) -> str:
    """Render the human-readable markdown report.

    Counts and bias names only - no raw private-corpus excerpts are emitted.
    """
    lines: list[str] = []
    lines.append("# Corpus Version Comparison")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}  ")
    lines.append(f"**Scenarios evaluated:** {len(scenarios)}  ")
    lines.append(f"**Versions compared:** {', '.join(per_version.keys())}")
    lines.append("")
    lines.append("> Counts and named biases only. Raw private-corpus excerpts are kept out of this report.")
    lines.append("")

    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | " + " | ".join(per_version.keys()) + " |")
    lines.append("|---|" + "|".join(["---"] * len(per_version)) + "|")

    metric_keys = [
        ("avg_biases_baseline", "Avg biases (baseline, no RAG)"),
        ("avg_biases_rag", "Avg biases (with RAG)"),
        ("avg_actions_baseline", "Avg recommended actions (baseline)"),
        ("avg_actions_rag", "Avg recommended actions (with RAG)"),
        ("scenarios_where_rag_added_biases", "Scenarios where RAG added biases"),
        ("avg_retrieval_count", "Avg retrieved chunks per query"),
    ]
    for key, label in metric_keys:
        row = [label]
        for version in per_version:
            row.append(str(per_version[version]["aggregate"].get(key, "n/a")))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Per-Scenario Highlights")
    lines.append("")
    for scenario in scenarios:
        lines.append(f"### Scenario {scenario['id']} - {scenario.get('domain', '')}")
        lines.append("")
        lines.append("> " + scenario["scenario"][:400] + ("..." if len(scenario["scenario"]) > 400 else ""))
        lines.append("")
        lines.append("| Version | Baseline biases | RAG biases | Added by RAG |")
        lines.append("|---|---|---|---|")
        for version, payload in per_version.items():
            record = next((r for r in payload["records"] if r["id"] == scenario["id"]), None)
            if not record:
                continue
            lines.append(
                f"| `{version}` | {', '.join(record['biases_baseline']) or '_none_'} | "
                f"{', '.join(record['biases_rag']) or '_none_'} | "
                f"{', '.join(record['added_by_rag']) or '_none_'} |"
            )
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. ``--versions`` is the only required argument."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--versions",
        required=True,
        help="Comma-separated list of corpus names to compare (e.g. 'public,private').",
    )
    parser.add_argument(
        "--scenario-ids",
        default=None,
        help="Optional comma-separated scenario ids; defaults to the full suite.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory (defaults to evaluation/).",
    )
    return parser.parse_args()


def main() -> None:
    """Run every selected scenario across every requested corpus version."""
    args = parse_args()
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    scenario_ids = [int(s) for s in args.scenario_ids.split(",")] if args.scenario_ids else None
    out_dir = Path(args.out_dir) if args.out_dir else get_evaluation_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    for version in versions:
        if not _corpus_exists(version):
            sys.exit(
                f"❌ Corpus '{version}' has no knowledge base. "
                f"Run `python data_pipeline/build_knowledge_base.py --corpus {version}` "
                f"and `python data_pipeline/build_vector_index.py --corpus {version}` first."
            )

    scenarios = _load_scenarios(scenario_ids)
    if not scenarios:
        sys.exit("No scenarios selected. Check --scenario-ids.")

    per_version: dict[str, dict] = {}

    for version in versions:
        print(f"\n=== Running {len(scenarios)} scenarios against corpus '{version}' ===")
        records = []
        raw_payloads = []
        for scenario in scenarios:
            print(f"  scenario {scenario['id']} ({scenario.get('domain', '')})")
            result = detect_bias_comparison(scenario["scenario"], corpus=version)
            records.append(_summarise_run(scenario, result))
            raw_payloads.append(
                {
                    "id": scenario["id"],
                    "scenario": scenario["scenario"],
                    "result": result,
                }
            )
        per_version[version] = {
            "records": records,
            "aggregate": _aggregate(records),
            "raw": raw_payloads,
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = out_dir / f"version_comparison_{timestamp}.json"
    md_path = out_dir / f"version_comparison_{timestamp}.md"

    with raw_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": timestamp,
                "scenario_ids": [s["id"] for s in scenarios],
                "versions": list(per_version.keys()),
                "results": per_version,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    md_path.write_text(_render_markdown(per_version, scenarios, timestamp), encoding="utf-8")

    print(f"\n✅ Wrote raw payloads to:    {raw_path}")
    print(f"✅ Wrote markdown summary to: {md_path}")


if __name__ == "__main__":
    main()
