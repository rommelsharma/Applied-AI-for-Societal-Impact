"""
Evaluation harness entry point.

Iterates over ``evaluation/test_scenarios.json``, runs the baseline-vs-RAG
comparison for every scenario, prints lightweight diff stats, and persists
the full per-scenario results to ``evaluation/latest_results.json``.

This script is the seed for richer scoring (grounding, retrieval relevance,
schema fidelity); for now it captures raw outputs that future scorecards will
operate on.
"""

import json
import sys
from pathlib import Path

# Walk one level up from this file to find the project root, then put it on
# sys.path so absolute imports (``app.services...``) resolve when the script
# is run directly from PyCharm or the terminal.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.services.bias_detector import compare_outputs, detect_bias_comparison


def main():
    """Run the evaluation suite end-to-end and persist results to disk."""
    scenarios_path = PROJECT_ROOT / "evaluation" / "test_scenarios.json"
    output_path = PROJECT_ROOT / "evaluation" / "latest_results.json"

    with scenarios_path.open("r", encoding="utf-8") as handle:
        scenarios = json.load(handle)

    results = []

    for scenario in scenarios:
        result = detect_bias_comparison(scenario["scenario"])
        compare_outputs(result["without_rag"], result["with_rag"])
        results.append(
            {
                "id": scenario["id"],
                "scenario": scenario["scenario"],
                "result": result,
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nSaved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
