"""Append-only log for ``data/eval/runs/sample_results_comparison.md`` (JSONL rows)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared_components.utilities.path_utils import get_legacy_response_dir, get_response_dir

SAMPLE_LOG_HEADER = """# Sample results comparison (append-only)

Each appended **line** is one logical row: a single JSON object with `timestamp` (ISO UTC),
then `without_rag` and `with_rag` (full model JSON payloads for baseline vs RAG paths).
Optional keys: `scenario_id`, `scenario_title`, `label` (from `record_response_run.py`).

Scripts append via `evaluation/sample_results_log.append_sample_results_jsonl` — run
`scripts/run_sample_comparison.py` or `scripts/record_response_run.py` after connectivity.

Outputs live under ``data/eval/runs/`` (legacy ``response/`` logs are migrated once).

---
"""


def migrate_docs_sample_if_present(project_root: Path) -> None:
    """Move legacy ``docs/sample_results_comparison.md`` into ``data/eval/runs/`` once."""
    docs_md = project_root / "docs" / "sample_results_comparison.md"
    if not docs_md.is_file():
        return
    dst = get_response_dir() / "sample_results_comparison.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    legacy = docs_md.read_text(encoding="utf-8")
    stamp = datetime.now(timezone.utc).isoformat()
    if dst.exists():
        with dst.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n\n<!-- migrated from docs/sample_results_comparison.md ({stamp}) -->\n\n"
            )
            handle.write(legacy)
    else:
        dst.write_text(
            SAMPLE_LOG_HEADER + "\n" + legacy + f"\n\n---\n<!-- migrated {stamp} -->\n",
            encoding="utf-8",
        )
    docs_md.unlink()


def migrate_legacy_response_dir() -> None:
    """Copy ``response/*`` logs into ``data/eval/runs`` once if the new tree is empty."""
    new_dir = get_response_dir()
    legacy = get_legacy_response_dir()
    new_dir.mkdir(parents=True, exist_ok=True)
    for name in ("sample_results_comparison.md", "connectivity_log.txt"):
        src = legacy / name
        dst = new_dir / name
        if not src.is_file() or dst.exists():
            continue
        dst.write_bytes(src.read_bytes())


def append_sample_results_jsonl(rows: list[dict[str, Any]], *, batch_timestamp: str | None = None) -> Path:
    """Append one JSON line per row to ``data/eval/runs/sample_results_comparison.md``."""
    migrate_legacy_response_dir()
    path = get_response_dir() / "sample_results_comparison.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(SAMPLE_LOG_HEADER + "\n", encoding="utf-8")
    ts = batch_timestamp or datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            payload = {"timestamp": ts, **row}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path
