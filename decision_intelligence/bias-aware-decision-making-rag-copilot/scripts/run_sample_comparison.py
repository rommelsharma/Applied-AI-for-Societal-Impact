"""
Run two illustrative scenarios through the baseline-vs-RAG bias detector.

Produces:
    * ``data/eval/runs/sample_results.json`` - raw payloads (without_rag, with_rag, retrieval).
    * ``data/eval/runs/sample_results_comparison.md`` - markdown header plus append-only JSONL
      lines (one JSON object per line).

Connectivity is exercised first (and logged to ``data/eval/runs/connectivity_log.txt`` via
``evaluation/connectivity.run_connectivity_check``).

Run from the project root:
    python scripts/run_sample_comparison.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.bias_detector import detect_bias_comparison
from evaluation.sample_results_log import append_sample_results_jsonl, migrate_docs_sample_if_present
from shared_components.utilities.path_utils import ensure_directory, get_response_dir


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-connectivity",
        action="store_true",
        help="Skip Bedrock + vector-store smoke checks (not recommended).",
    )
    args = parser.parse_args()

    migrate_docs_sample_if_present(PROJECT_ROOT)

    if not args.skip_connectivity:
        from evaluation.connectivity import run_connectivity_check

        connectivity = run_connectivity_check()
        if not connectivity.get("ok"):
            print(json.dumps(connectivity, indent=2), file=sys.stderr)
            sys.exit(1)

    results: list[dict] = []
    batch_ts = datetime.now(timezone.utc).isoformat()

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

    json_output = ensure_directory(get_response_dir()) / "sample_results.json"
    with json_output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved raw results to: {json_output}")

    jsonl_rows: list[dict] = []
    for entry in results:
        jsonl_rows.append(
            {
                "scenario_id": entry["id"],
                "scenario_title": entry["title"],
                "without_rag": entry["result"]["without_rag"],
                "with_rag": entry["result"]["with_rag"],
            }
        )
    log_path = append_sample_results_jsonl(jsonl_rows, batch_timestamp=batch_ts)
    print(f"Appended JSONL rows to: {log_path}")


if __name__ == "__main__":
    main()
