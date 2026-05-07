"""
Interactive demo entry point.

Runs the same scenarios that the evaluation harness uses but prints the full
baseline + RAG + retrieval payloads, which is helpful for visual inspection in
PyCharm or a notebook.
"""

import json
import sys
from pathlib import Path

# Make the project root importable so this script works as a "Run File" target
# without needing a packaged install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.services.bias_detector import compare_outputs, detect_bias_comparison, print_comparison


def run_demo():
    """Iterate every test scenario and print full comparison output for each."""
    scenarios_path = PROJECT_ROOT / "evaluation" / "test_scenarios.json"
    with scenarios_path.open("r", encoding="utf-8") as handle:
        scenarios = json.load(handle)

    for scenario in scenarios:
        print("\nSCENARIO:", scenario["scenario"])
        result = detect_bias_comparison(scenario["scenario"])
        print_comparison(result)
        compare_outputs(result["without_rag"], result["with_rag"])


if __name__ == "__main__":
    run_demo()
