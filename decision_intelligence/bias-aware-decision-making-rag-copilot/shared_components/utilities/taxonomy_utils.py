"""
Helpers for loading taxonomy metadata used by retrieval and prompting.
"""

from __future__ import annotations

import json
from functools import lru_cache

from shared_components.utilities.path_utils import get_metadata_dir


def _load_json_file(file_name: str) -> list[dict]:
    file_path = get_metadata_dir() / file_name
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_bias_taxonomy() -> list[dict]:
    return _load_json_file("bias-taxonomy.json")


@lru_cache(maxsize=1)
def load_support_concepts() -> list[dict]:
    return _load_json_file("retrieval-concepts.json")


@lru_cache(maxsize=1)
def build_concept_catalog() -> dict[str, dict]:
    catalog: dict[str, dict] = {}

    for entry in load_bias_taxonomy():
        aliases = entry.get("aliases", [])
        catalog[entry["bias_name"]] = {
            "name": entry["bias_name"],
            "kind": "bias",
            "aliases": aliases,
            "decision_domains": entry.get("decision_domains", []),
            "importance": entry.get("retrieval_importance", "medium"),
        }

    for entry in load_support_concepts():
        aliases = entry.get("aliases", [])
        catalog[entry["concept_name"]] = {
            "name": entry["concept_name"],
            "kind": "support_concept",
            "aliases": aliases,
            "decision_domains": entry.get("decision_domains", []),
            "importance": entry.get("retrieval_importance", "medium"),
        }

    return catalog


def normalize_phrase(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())
