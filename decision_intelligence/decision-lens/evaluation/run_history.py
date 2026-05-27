"""
UI run history — save, prune, and reload the last N scenario runs.

Files are stored as individual timestamped JSON documents in
``data/eval/runs/`` alongside the existing eval captures.

File naming
-----------
``{YYYYMMDD_HHMMSS}_ui_run.json``

Example: ``20260516_143022_ui_run.json``

The ``_ui_run`` suffix distinguishes these from CLI-generated captures
(``*_rag_eval.json``, ``*_before_refactor.json``, etc.) so the two sets
can coexist without collision.

Each file schema
----------------
{
    "saved_at":             ISO-8601 UTC timestamp,
    "scenario":             full scenario text,
    "scenario_preview":     first 120 chars (for list displays),
    "corpus":               "public" | "private",
    "top_k":                int,
    "mmr_lambda":           float,
    "bias_count_baseline":  int,
    "bias_count_rag":       int,
    "bias_lift":            int   (rag minus baseline),
    "groundedness_score":   float | null,
    "retrieval_gap":        bool,
    "chunks_retrieved":     int,
    "result":               full detect_bias_comparison() return dict
}

Pruning
-------
After every save, runs beyond MAX_UI_RUNS (default 20) are deleted
oldest-first so the folder never accumulates unboundedly.

Public API
----------
    save_ui_run(scenario, corpus, top_k, mmr_lambda, result) -> Path
    load_recent_runs(n=MAX_UI_RUNS) -> list[dict]   # lightweight index (no result blob)
    load_run(path) -> dict                           # full JSON for one run
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared_components.utilities.path_utils import get_response_dir

# Maximum number of UI run files retained on disk.
MAX_UI_RUNS: int = 20

# Glob pattern that identifies UI-generated run files (never matches CLI evals).
_UI_RUN_GLOB: str = "*_ui_run.json"


# ── internal helpers ──────────────────────────────────────────────────────────

def _runs_dir() -> Path:
    d = get_response_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sorted_ui_run_files(directory: Path) -> list[Path]:
    """Return ui_run files sorted oldest → newest by filename (timestamp prefix)."""
    return sorted(directory.glob(_UI_RUN_GLOB))


def _prune(directory: Path, max_runs: int = MAX_UI_RUNS) -> list[Path]:
    """Delete the oldest ui_run files when the total exceeds *max_runs*.

    Returns the list of deleted paths.
    """
    files = _sorted_ui_run_files(directory)
    excess = files[: max(0, len(files) - max_runs)]
    for f in excess:
        try:
            f.unlink(missing_ok=True)
        except OSError:
            pass
    return excess


def _extract_summary(scenario: str, result: dict) -> dict:
    """Pull lightweight scalar summary fields out of a full result dict."""
    without = result.get("without_rag") or {}
    with_rag = result.get("with_rag") or {}
    prov = result.get("retrieval_provenance") or {}

    bc_base = len(without.get("biases_identified") or [])
    bc_rag = len(with_rag.get("biases_identified") or [])

    return {
        "scenario_preview": scenario[:120].rstrip() + ("…" if len(scenario) > 120 else ""),
        "bias_count_baseline": bc_base,
        "bias_count_rag": bc_rag,
        "bias_lift": bc_rag - bc_base,
        "groundedness_score": prov.get("groundedness_score"),
        "retrieval_gap": prov.get("retrieval_gap", False),
        "chunks_retrieved": prov.get("chunks_retrieved", 0),
    }


# ── public API ────────────────────────────────────────────────────────────────

def save_ui_run(
    scenario: str,
    corpus: str,
    top_k: int,
    mmr_lambda: float,
    result: dict,
    *,
    max_runs: int = MAX_UI_RUNS,
) -> Path:
    """Persist one UI run to ``data/eval/runs/`` and prune older files.

    Returns the path of the newly written file.
    """
    directory = _runs_dir()
    ts = datetime.now(timezone.utc)
    # Include microseconds so rapid successive saves never collide on the filename.
    ts_str = ts.strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{ts_str}_ui_run.json"
    file_path = directory / filename

    summary = _extract_summary(scenario, result)

    payload = {
        "saved_at": ts.isoformat(),
        "scenario": scenario,
        **summary,
        "corpus": corpus,
        "top_k": top_k,
        "mmr_lambda": mmr_lambda,
        "result": result,
    }

    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _prune(directory, max_runs=max_runs)
    return file_path


def load_recent_runs(n: int = MAX_UI_RUNS) -> list[dict]:
    """Return index records for the *n* most recent UI runs, newest first.

    Each record contains all scalar fields but omits the large ``result``
    blob to keep the sidebar load fast. Use :func:`load_run` to fetch the
    full payload for a specific file.
    """
    directory = _runs_dir()
    files = _sorted_ui_run_files(directory)
    recent = list(reversed(files))[:n]

    records: list[dict] = []
    for path in recent:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                {
                    "file": path.name,
                    "path": str(path),
                    "saved_at": data.get("saved_at", ""),
                    "scenario_preview": data.get("scenario_preview", ""),
                    "corpus": data.get("corpus", ""),
                    "top_k": data.get("top_k"),
                    "mmr_lambda": data.get("mmr_lambda"),
                    "bias_count_baseline": data.get("bias_count_baseline"),
                    "bias_count_rag": data.get("bias_count_rag"),
                    "bias_lift": data.get("bias_lift"),
                    "groundedness_score": data.get("groundedness_score"),
                    "retrieval_gap": data.get("retrieval_gap", False),
                    "chunks_retrieved": data.get("chunks_retrieved", 0),
                }
            )
        except Exception:
            # Silently skip unreadable files — never break the UI
            continue

    return records


def load_run(path: str | Path) -> dict:
    """Load the full payload (including ``result``) for one saved run file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
