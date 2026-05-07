import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.services.bias_detector import compare_outputs, detect_bias_comparison


def main():
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
