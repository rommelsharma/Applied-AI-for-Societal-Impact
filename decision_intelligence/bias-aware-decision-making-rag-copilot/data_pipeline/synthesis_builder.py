"""Build ``processed/enriched/synthesis.json`` (heuristic, no extra LLM calls by default)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from data_pipeline.bootstrap import add_project_root_to_sys_path
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_pipeline.bootstrap import add_project_root_to_sys_path

add_project_root_to_sys_path()

from shared_components.utilities.path_utils import (
    ensure_directory,
    get_corpus_name,
    resolve_knowledge_base_json,
    resolve_synthesis_json,
)
from shared_components.utilities.taxonomy_utils import get_ontology_version


def build_synthesis(corpus: str | None = None) -> Path:
    kb_path = resolve_knowledge_base_json(corpus)
    with kb_path.open("r", encoding="utf-8") as handle:
        kb = json.load(handle)

    by_source: dict[str, list[dict]] = {}
    for row in kb:
        if not isinstance(row, dict):
            continue
        src = row.get("source") or row.get("file_name") or "unknown"
        by_source.setdefault(str(src), []).append(row)

    books: dict[str, dict] = {}
    for slug, rows in by_source.items():
        concepts_union: list[str] = sorted({c for r in rows for c in r.get("concepts", []) if isinstance(c, str)})
        iv_union: list[str] = sorted(
            {i for r in rows for i in r.get("interventions", []) if isinstance(i, str)}
        )
        fm_union: list[str] = sorted(
            {f for r in rows for f in r.get("failure_modes", []) if isinstance(f, str)}
        )
        summaries = [r.get("summary") or "" for r in rows[:5]]
        thesis = " ".join(s for s in summaries if s).strip()[:1200]
        books[slug] = {
            "core_thesis": thesis or f"Curated excerpts from {slug}.",
            "main_decision_failures": fm_union[:12],
            "unique_concepts": concepts_union[:50],
            "key_interventions": iv_union[:20],
            "best_empirical_findings": [],
            "chunk_count": len(rows),
        }

    # Cross-book: top overlapping concepts across corpus
    concept_counts: Counter[str] = Counter()
    for row in kb:
        if not isinstance(row, dict):
            continue
        for c in row.get("concepts", []) or []:
            if isinstance(c, str):
                concept_counts[c] += 1
    cross_book = [
        {
            "concept": concept,
            "matched_concepts": [concept],
            "convergent_findings": f"Mentioned in {count} chunks corpus-wide.",
            "divergent_findings": "",
            "recommended_interventions": [],
        }
        for concept, count in concept_counts.most_common(12)
    ]

    doc = {
        "ontology_version": get_ontology_version(),
        "generator": "synthesis_builder_heuristic_v1",
        "books": books,
        "chapters": {},
        "cross_book": cross_book,
    }

    out = resolve_synthesis_json(corpus)
    ensure_directory(out.parent)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=2)
    print(f"Wrote synthesis to {out}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=None)
    return p.parse_args()


if __name__ == "__main__":
    build_synthesis(corpus=parse_args().corpus)
