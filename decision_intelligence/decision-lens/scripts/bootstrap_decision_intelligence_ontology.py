#!/usr/bin/env python3
"""Build data/registry/decision_intelligence_ontology.json from legacy metadata JSON.

Run from repo root:
    python scripts/bootstrap_decision_intelligence_ontology.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "registry" / "decision_intelligence_ontology.json"
META = ROOT / "data" / "metadata"


def main() -> None:
    with (META / "bias-taxonomy.json").open(encoding="utf-8") as f:
        biases = json.load(f)
    with (META / "retrieval-concepts.json").open(encoding="utf-8") as f:
        support = json.load(f)

    # Reverse map: bias_name -> support concepts that list it in related_biases
    mitigators: dict[str, list[str]] = {}
    for row in support:
        cname = row.get("concept_name")
        if not isinstance(cname, str):
            continue
        for rb in row.get("related_biases") or []:
            if isinstance(rb, str):
                mitigators.setdefault(rb, []).append(cname)
    for k in mitigators:
        mitigators[k] = sorted(set(mitigators[k]))

    concepts: dict[str, dict] = {}
    for entry in biases:
        cid = entry.get("bias_name")
        if not isinstance(cid, str):
            continue
        related = [x for x in (entry.get("related_biases") or []) if isinstance(x, str)]
        concepts[cid] = {
            "concept": cid,
            "kind": "cognitive_mechanism",
            "aliases": list(entry.get("aliases") or []),
            "related_concepts": sorted(set(related)),
            "mitigated_by": mitigators.get(cid, []),
            "failure_modes": [],
            "group_dynamics": [],
            "decision_phases": [],
            "interventions": [],
            "evidence_type": [],
            "decision_domains": sorted({x for x in (entry.get("decision_domains") or []) if isinstance(x, str)}),
            "source_books": list(entry.get("source") or []),
            "importance": entry.get("retrieval_importance", "medium"),
        }

    for entry in support:
        cid = entry.get("concept_name")
        if not isinstance(cid, str) or cid in concepts:
            continue
        related = [x for x in (entry.get("related_biases") or []) if isinstance(x, str)]
        concepts[cid] = {
            "concept": cid,
            "kind": "intervention" if "interview" in cid or "audit" in cid or "outside" in cid else "framework",
            "aliases": list(entry.get("aliases") or []),
            "related_concepts": sorted(set(related)),
            "mitigated_by": [],
            "failure_modes": [],
            "group_dynamics": [],
            "decision_phases": [],
            "interventions": [cid] if "structured" in cid or "interview" in cid else [],
            "evidence_type": [],
            "decision_domains": sorted({x for x in (entry.get("decision_domains") or []) if isinstance(x, str)}),
            "source_books": [],
            "importance": entry.get("retrieval_importance", "medium"),
        }

    doc = {
        "ontology_version": "1.0.0",
        "description": "Bootstrapped from bias-taxonomy.json and retrieval-concepts.json; curate graph edges over time.",
        "concepts": concepts,
    }
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"Wrote {REGISTRY} with {len(concepts)} concepts.")


if __name__ == "__main__":
    main()
