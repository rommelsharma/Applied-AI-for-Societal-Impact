"""
Interactive demo: run every extended-suite scenario through the bias detector.

Prints full baseline + RAG + retrieval payloads for local inspection (IDE or terminal).
Not an .ipynb notebook — run as a script: ``python examples/run_bias_scenarios_demo.py``.
"""

import sys
from pathlib import Path

# Project root is parent of ``examples/`` — append for runnable-script imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.services.bias_detector import compare_outputs, detect_bias_comparison, print_comparison
from evaluation.scenario_catalog import get_extended_test_scenarios


def run_demo():
    """Iterate every test scenario and print full comparison output for each."""
    scenarios = get_extended_test_scenarios()

    for scenario in scenarios:
        print("\nSCENARIO:", scenario["scenario"])
        result = detect_bias_comparison(scenario["scenario"])
        print_comparison(result)
        compare_outputs(result["without_rag"], result["with_rag"])


if __name__ == "__main__":
    run_demo()
