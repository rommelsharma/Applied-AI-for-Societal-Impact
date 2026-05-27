#!/usr/bin/env python3
"""Merge scenario JSON files into ``data/eval/gold/scenarios_catalog.json``.

Source files (baseline, extended suite, private-book prompts, gold labels) live
under ``data/eval/gold/``. Older copies may still exist under ``evaluation/`` or
a locally recreated ``archived/evaluation_jsonsources/`` — those paths are used only as fallbacks.

    python scripts/build_scenarios_catalog.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared_components.utilities.path_utils import ensure_directory, get_eval_gold_dir


def _read_json(path: Path) -> list | dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    gold = get_eval_gold_dir()
    ensure_directory(gold)

    arch = PROJECT_ROOT / "archived" / "evaluation_jsonsources"

    def pick(name: str) -> Path:
        """Prefer committed gold dir, then archived fallbacks, then evaluation/."""
        p_gold = gold / name
        p_arch = arch / name
        p_eval = PROJECT_ROOT / "evaluation" / name
        if p_gold.is_file():
            return p_gold
        if p_arch.is_file():
            return p_arch
        return p_eval

    baseline_path = pick("baseline_scenarios.json")
    extended_path = pick("test_scenarios.json")
    private_path = pick("private_book_scenario_questions.json")
    gold_labels_path = gold / "gold_labels.json"
    if not gold_labels_path.is_file():
        gold_labels_path = pick("gold_labels.json")

    baseline = _read_json(baseline_path) if baseline_path.is_file() else []
    extended = _read_json(extended_path) if extended_path.is_file() else []
    private_raw = _read_json(private_path) if private_path.is_file() else []

    gold_expectations: dict = {}
    if gold_labels_path.is_file():
        gl = _read_json(gold_labels_path)
        if isinstance(gl, dict) and isinstance(gl.get("scenarios"), dict):
            gold_expectations = {"scenarios": gl["scenarios"]}
        elif isinstance(gl, dict):
            gold_expectations = gl

    private_norm: list[dict] = []
    for row in private_raw:
        if not isinstance(row, dict):
            continue
        q = row.get("question") or ""
        out = {**row, "scenario": q, "suite": "private_book"}
        if "title" not in out and out.get("source_book"):
            out["title"] = f"{out['source_book']} ({out.get('id', '')})"
        private_norm.append(out)

    catalog = {
        "catalog_version": "2.0.0",
        "description": "Canonical eval scenarios: baseline freeze, extended suite, private-corpus book prompts, gold expectations.",
        "baseline": baseline,
        "extended_suite": extended,
        "private_book_questions": private_norm,
        "gold_expectations": gold_expectations,
    }

    out_path = gold / "scenarios_catalog.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(catalog, handle, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
