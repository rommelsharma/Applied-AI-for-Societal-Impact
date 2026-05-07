"""
Taxonomy loaders shared by the indexing pipeline and the runtime layer.

The bias taxonomy and the support-concept catalog are both data-driven JSON
files, kept outside code so they can be edited without redeploying. This
module wraps both:
    * ``load_bias_taxonomy``    - cached load of the bias entries.
    * ``load_support_concepts`` - cached load of the support-concept entries.
    * ``build_concept_catalog`` - merges both into a single dict that
      ``concept_extractor`` and ``enrich_chunks`` consume.

``lru_cache`` is used so the JSON files are read at most once per process.
"""

from __future__ import annotations

import json
from functools import lru_cache

from shared_components.utilities.path_utils import get_metadata_dir


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
def build_concept_catalog() -> dict[str, dict]:
    """Merge biases and support concepts into one alias-rich lookup dict.

    Each entry exposes the same fields regardless of whether it originated as a
    bias or a support concept (``name``, ``kind``, ``aliases``,
    ``decision_domains``, ``importance``), which is what lets the rest of the
    code treat them uniformly.

    Cached because it is called from hot paths (per-chunk and per-query
    concept extraction) and the underlying JSON is immutable at runtime.
    """
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
    """Normalise an alias/canonical phrase for matching.

    Lowercases, replaces underscores and hyphens with spaces, and collapses
    whitespace. Used on both sides of the comparison in
    ``concept_extractor.extract_concepts`` to ensure consistent matching.
    """
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())
