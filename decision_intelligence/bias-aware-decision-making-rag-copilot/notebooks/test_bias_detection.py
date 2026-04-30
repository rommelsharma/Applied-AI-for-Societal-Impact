import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.bias_detector import detect_bias_comparison, print_comparison

import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
file_path = os.path.join(BASE_DIR, "evaluation", "test_scenarios.json")

with open(file_path) as f:
    scenarios = json.load(f)
count = 0
for s in scenarios:
    if count > 0:
        print("Test phase, only 1 scenario being tests")
        break
    print("\nSCENARIO:", s["scenario"])
    result = detect_bias_comparison(s["scenario"])
    print_comparison(result)
    count += 1