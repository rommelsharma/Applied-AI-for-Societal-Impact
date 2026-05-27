"""Load the canonical scenarios catalog under ``data/eval/gold/scenarios_catalog.json``."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from shared_components.utilities.path_utils import get_eval_gold_dir


def scenarios_catalog_path() -> Path:
    return get_eval_gold_dir() / "scenarios_catalog.json"


@lru_cache(maxsize=1)
def load_scenarios_catalog() -> dict:
    path = scenarios_catalog_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing scenarios catalog: {path}. "
            "Run `python scripts/build_scenarios_catalog.py` after checkout, or restore "
            "`data/eval/gold/` scenario JSON sources and rebuild."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reload_scenarios_catalog() -> dict:
    """Clear cache (e.g. after tests edit the file on disk)."""
    load_scenarios_catalog.cache_clear()
    return load_scenarios_catalog()


def get_baseline_scenarios() -> list[dict]:
    return list(load_scenarios_catalog().get("baseline") or [])


def get_extended_test_scenarios() -> list[dict]:
    return list(load_scenarios_catalog().get("extended_suite") or [])


def get_private_book_questions() -> list[dict]:
    return list(load_scenarios_catalog().get("private_book_questions") or [])


def get_gold_expectations() -> dict:
    """Per-scenario eval hints: failure modes, interventions, relevant concepts."""
    raw = load_scenarios_catalog().get("gold_expectations") or {}
    if isinstance(raw, dict) and "scenarios" in raw and isinstance(raw["scenarios"], dict):
        return raw["scenarios"]
    return raw if isinstance(raw, dict) else {}


def scenario_text_for_detection(entry: dict) -> str:
    """Text passed to ``detect_bias_comparison`` (extended suite uses ``scenario``; private book uses ``question``)."""
    s = entry.get("scenario")
    if isinstance(s, str) and s.strip():
        return s.strip()
    q = entry.get("question")
    if isinstance(q, str) and q.strip():
        return q.strip()
    return ""


def scenario_title(entry: dict) -> str:
    if isinstance(entry.get("title"), str) and entry["title"].strip():
        return entry["title"].strip()
    if isinstance(entry.get("source_book"), str) and entry["source_book"].strip():
        return f"{entry['source_book']} — {entry.get('id', '')}"
    return str(entry.get("id", "scenario"))
