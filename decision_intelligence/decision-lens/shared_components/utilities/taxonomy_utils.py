"""
Taxonomy and ontology loaders shared by the indexing pipeline and runtime.

* ``load_decision_intelligence_ontology`` — cached ``data/registry/decision_intelligence_ontology.json``.
* ``build_concept_catalog`` — alias lookup for concept extraction (ontology-first; legacy fallback).
* ``normalize_phrase`` — shared normalisation for substring matching.
"""

from __future__ import annotations

import json
from functools import lru_cache

from shared_components.utilities.path_utils import get_metadata_dir, get_registry_dir


def _load_json_file(file_name: str) -> list[dict]:
    """Read a metadata JSON file relative to ``data/metadata/``."""
    file_path = get_metadata_dir() / file_name
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_bias_taxonomy() -> list[dict]:
    """Return the full bias taxonomy as a list of entries (cached)."""
    return _load_json_file("bias-taxonomy.json")


@lru_cache(maxsize=1)
def load_support_concepts() -> list[dict]:
    """Return the support-concept catalog (e.g. ``decision_hygiene``, ``noise_audit``)."""
    return _load_json_file("retrieval-concepts.json")


@lru_cache(maxsize=1)
def load_decision_intelligence_ontology() -> dict:
    """Load the v4 ontology brain (concepts + graph metadata)."""
    path = get_registry_dir() / "decision_intelligence_ontology.json"
    if not path.is_file():
        return {"ontology_version": "0", "concepts": {}}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_ontology_version() -> str:
    """Return ontology_version string for run cards and manifests."""
    return str(load_decision_intelligence_ontology().get("ontology_version") or "0")


@lru_cache(maxsize=1)
def build_concept_catalog() -> dict[str, dict]:
    """Merge ontology concepts into one alias-rich lookup dict.

    If ``decision_intelligence_ontology.json`` defines ``concepts``, those
    become the catalog. Otherwise falls back to merging legacy JSON files.
    """
    onto = load_decision_intelligence_ontology()
    cmap = onto.get("concepts") or {}
    if isinstance(cmap, dict) and cmap:
        catalog: dict[str, dict] = {}
        for cid, node in cmap.items():
            if not isinstance(cid, str) or not isinstance(node, dict):
                continue
            catalog[cid] = {
                "name": cid,
                "kind": node.get("kind", "concept"),
                "aliases": list(node.get("aliases") or []),
                "decision_domains": list(node.get("decision_domains") or []),
                "importance": node.get("importance", "medium"),
            }
        return catalog

    catalog = {}
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
    """Normalise an alias/canonical phrase for matching."""
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())
