"""Book / chapter / cross-book synthesis selection for hierarchical context."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_synthesis(path: Path) -> dict[str, Any]:
    """Load ``synthesis.json``; return empty structure if missing."""
    if not path.is_file():
        return {"books": {}, "chapters": {}, "cross_book": []}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_synthesis_blocks(
    synthesis: dict[str, Any],
    concepts: list[str],
    *,
    max_books: int = 2,
    max_cross: int = 2,
) -> list[dict[str, Any]]:
    """Pick high-level synthesis rows overlapping query ``concepts``."""
    concept_set = {c for c in concepts if isinstance(c, str)}
    blocks: list[dict[str, Any]] = []

    books = synthesis.get("books") or {}
    if isinstance(books, dict):
        for slug, row in books.items():
            if not isinstance(row, dict):
                continue
            uniq = set(row.get("unique_concepts") or [])
            if concept_set and not (uniq & concept_set):
                continue
            blocks.append({"level": "book", "source": slug, "content": row})
            if len([b for b in blocks if b["level"] == "book"]) >= max_books:
                break

    cross = synthesis.get("cross_book") or []
    if isinstance(cross, list):
        for row in cross:
            if len([b for b in blocks if b["level"] == "cross_book"]) >= max_cross:
                break
            if not isinstance(row, dict):
                continue
            matched = set(row.get("matched_concepts") or [])
            if concept_set and not (matched & concept_set):
                continue
            blocks.append({"level": "cross_book", "content": row})

    return blocks


def format_synthesis_blocks(blocks: list[dict[str, Any]]) -> str:
    """Render synthesis picks as prompt-ready text."""
    if not blocks:
        return ""
    parts: list[str] = []
    for i, b in enumerate(blocks, start=1):
        level = b.get("level", "book")
        content = b.get("content") or {}
        if level == "book":
            src = b.get("source", "")
            thesis = content.get("core_thesis") or ""
            iv = content.get("key_interventions") or []
            parts.append(f"[Synthesis {i} — book:{src}]\n{thesis}\nInterventions: {', '.join(iv) if iv else '—'}")
        else:
            parts.append(f"[Synthesis {i} — cross_book]\n{json.dumps(content, indent=2)[:1200]}")
    return "\n\n".join(parts)
