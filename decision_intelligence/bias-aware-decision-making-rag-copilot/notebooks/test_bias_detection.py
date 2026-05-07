import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.services.bias_detector import compare_outputs, detect_bias_comparison, print_comparison


def run_demo():
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
