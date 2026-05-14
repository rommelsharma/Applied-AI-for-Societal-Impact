#!/usr/bin/env python3
"""Connectivity check + fixed scenarios → timestamped JSON under ``response/``.

Used to capture **before_refactor** baselines and later runs (e.g. after hybrid
RAG) so the same three scenarios can be compared over time. Each file includes
a **run_card** (Phase 4) and **metrics** (Phase 1).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        default="before_refactor",
        help="Short tag embedded in the filename (e.g. before_refactor, after_phase5_hybrid).",
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Path to scenarios JSON (default: evaluation/baseline_scenarios.json).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for JSON output (default: <project>/response).",
    )
    parser.add_argument(
        "--skip-connectivity",
        action="store_true",
        help="Skip Bedrock/vector checks (not recommended for baseline capture).",
    )
    parser.add_argument(
        "--no-append-sample-log",
        dest="append_sample_log",
        action="store_false",
        default=True,
        help="Do not append JSONL rows to response/sample_results_comparison.md (default: append).",
    )
    args = parser.parse_args()

    scenario_path = Path(args.scenarios) if args.scenarios else PROJECT_ROOT / "evaluation" / "baseline_scenarios.json"
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "response"
    out_dir.mkdir(parents=True, exist_ok=True)

    from app.services.bias_detector import detect_bias_comparison, load_taxonomy
    from evaluation.connectivity import run_connectivity_check
    from evaluation.metrics import aggregate_run_metrics, score_comparison_result
    from evaluation.run_card import build_run_card
    from evaluation.sample_results_log import append_sample_results_jsonl, migrate_docs_sample_if_present

    if args.skip_connectivity:
        connectivity: dict = {"skipped": True}
    else:
        connectivity = run_connectivity_check()
        if not connectivity.get("ok"):
            print(json.dumps(connectivity, indent=2), file=sys.stderr)
            sys.exit(1)

    if not scenario_path.is_file():
        print(f"Scenarios file not found: {scenario_path}", file=sys.stderr)
        sys.exit(1)

    with scenario_path.open("r", encoding="utf-8") as handle:
        scenarios = json.load(handle)

    taxonomy = load_taxonomy()
    rows: list[dict] = []

    for spec in scenarios:
        t0 = time.perf_counter()
        result = detect_bias_comparison(spec["scenario"])
        elapsed = time.perf_counter() - t0
        metrics = score_comparison_result(result, taxonomy=taxonomy)
        rows.append(
            {
                "scenario_id": spec.get("id"),
                "title": spec.get("title"),
                "domain": spec.get("domain"),
                "scenario_text": spec["scenario"],
                "latency_seconds": round(elapsed, 3),
                "result": result,
                "metrics": metrics,
            }
        )

    aggregate = aggregate_run_metrics([r["metrics"] for r in rows])
    run_card = build_run_card(
        project_root=PROJECT_ROOT,
        scenario_file=scenario_path.resolve(),
        label=args.label,
        extra={"scenario_count": len(scenarios), "script": "record_response_run.py"},
    )

    payload = {
        "connectivity": connectivity,
        "run_card": run_card,
        "scenarios": rows,
        "aggregate_metrics": aggregate,
    }

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_name = f"{ts}_{args.label}_rag_eval.json"
    out_path = out_dir / out_name
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(out_path)

    if args.append_sample_log:
        migrate_docs_sample_if_present(PROJECT_ROOT)
        batch_ts = datetime.now(timezone.utc).isoformat()
        jsonl_rows = [
            {
                "scenario_id": r.get("scenario_id") or r.get("title") or "unknown",
                "scenario_title": r.get("title"),
                "label": args.label,
                "without_rag": r["result"]["without_rag"],
                "with_rag": r["result"]["with_rag"],
            }
            for r in rows
        ]
        log_path = append_sample_results_jsonl(jsonl_rows, batch_timestamp=batch_ts)
        print(log_path)


if __name__ == "__main__":
    main()
