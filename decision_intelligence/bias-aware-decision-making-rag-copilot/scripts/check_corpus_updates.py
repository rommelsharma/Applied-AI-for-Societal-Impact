"""
Corpus freshness checker and incremental ingest trigger.

Compares the source files currently on disk against a persistent
``corpus_fingerprint.json`` to detect new, changed, or removed documents.
Optionally re-runs the full knowledge-base pipeline and rebuilds the
vector index for any affected corpus.

Fingerprint file
----------------
``data/corpora/<corpus>/corpus_fingerprint.json`` — written/updated after
every successful ingest so the next check can diff against it.

What is tracked
---------------
- ``parsed_text/*.txt``  — synthesised dossiers and any author-content files
  produced by ``scripts/ingest_author_content.py``.  These are the direct
  input to the chunker, so a change here always warrants a re-index.
- ``raw/*.pdf``          — raw source PDFs (private corpus).  Detected as
  "unparsed" when a corresponding ``parsed_text/<stem>.txt`` is absent.

Usage
-----
    # Report only (no Bedrock calls, safe to run anytime)
    python scripts/check_corpus_updates.py
    python scripts/check_corpus_updates.py --corpus private

    # Report and run full pipeline for the affected corpus
    python scripts/check_corpus_updates.py --ingest
    python scripts/check_corpus_updates.py --corpus private --ingest

    # Force a full rebuild even when nothing changed (e.g. after editing .env)
    python scripts/check_corpus_updates.py --ingest --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared_components.utilities.path_utils import (
    get_corpus_dir,
    get_corpus_name,
    get_parsed_text_dir,
    get_raw_data_dir,
)

# ── constants ─────────────────────────────────────────────────────────────────
FINGERPRINT_FILENAME = "corpus_fingerprint.json"


# ── hashing ───────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65_536), b""):
            h.update(block)
    return h.hexdigest()


def _file_record(path: Path) -> dict:
    return {
        "file": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


# ── fingerprint I/O ───────────────────────────────────────────────────────────

def _fingerprint_path(corpus: str) -> Path:
    return get_corpus_dir(corpus) / FINGERPRINT_FILENAME


def _load_fingerprint(corpus: str) -> dict:
    fp = _fingerprint_path(corpus)
    if fp.is_file():
        return json.loads(fp.read_text(encoding="utf-8"))
    return {"corpus": corpus, "sources": {}, "unparsed_pdfs": []}


def _save_fingerprint(corpus: str, sources: dict[str, dict], unparsed_pdfs: list[str]) -> None:
    data = {
        "corpus": corpus,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_count": len(sources),
        "sources": sources,
        "unparsed_pdfs": unparsed_pdfs,
    }
    _fingerprint_path(corpus).write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


# ── scanning ──────────────────────────────────────────────────────────────────

def _scan_parsed_texts(corpus: str) -> dict[str, dict]:
    """Hash every .txt file in parsed_text/ — these feed the chunker."""
    directory = get_parsed_text_dir(corpus)
    if not directory.is_dir():
        return {}
    return {p.stem: _file_record(p) for p in sorted(directory.glob("*.txt"))}


def _scan_raw_pdfs(corpus: str) -> list[str]:
    """Return PDF stems in raw/ that have no matching parsed_text/ .txt."""
    raw_dir = get_raw_data_dir(corpus)
    parsed_dir = get_parsed_text_dir(corpus)
    if not raw_dir.is_dir():
        return []
    parsed_stems = {p.stem for p in parsed_dir.glob("*.txt")} if parsed_dir.is_dir() else set()
    return [p.stem for p in sorted(raw_dir.glob("*.pdf")) if p.stem not in parsed_stems]


# ── diff ──────────────────────────────────────────────────────────────────────

def _diff(current: dict[str, dict], known: dict[str, dict]) -> dict:
    new_files = sorted(s for s in current if s not in known)
    changed = sorted(
        s for s in current
        if s in known and current[s]["sha256"] != known[s]["sha256"]
    )
    removed = sorted(s for s in known if s not in current)
    return {
        "up_to_date": not new_files and not changed and not removed,
        "new_files": new_files,
        "changed_files": changed,
        "removed_files": removed,
    }


# ── reporting ─────────────────────────────────────────────────────────────────

def _print_report(corpus: str, diff: dict, unparsed_pdfs: list[str]) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Corpus freshness check: {corpus!r}")
    print(sep)

    if diff["up_to_date"] and not unparsed_pdfs:
        print("  ✅  Index is up to date. No action needed.")
        print(sep)
        return

    if diff["new_files"]:
        print(f"\n  🆕  New text files (not yet indexed): {len(diff['new_files'])}")
        for f in diff["new_files"]:
            print(f"        + {f}.txt")

    if diff["changed_files"]:
        print(f"\n  ✏️   Changed text files (content hash differs): {len(diff['changed_files'])}")
        for f in diff["changed_files"]:
            print(f"        ~ {f}.txt")

    if diff["removed_files"]:
        print(f"\n  🗑️   Removed text files (still in last fingerprint): {len(diff['removed_files'])}")
        for f in diff["removed_files"]:
            print(f"        - {f}.txt")

    if unparsed_pdfs:
        print(f"\n  📄  Raw PDFs not yet parsed into text: {len(unparsed_pdfs)}")
        for f in unparsed_pdfs:
            print(f"        ? {f}.pdf")

    print(f"\n  ⚠️   Run with --ingest to rebuild the knowledge base and index.")
    print(sep)


# ── pipeline runner ────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> None:
    """Run a pipeline step as a subprocess, streaming output, raise on failure."""
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def _ingest(corpus: str) -> None:
    python = sys.executable
    print(f"\n{'='*60}")
    print(f"  Running full pipeline for corpus: {corpus!r}")
    print(f"{'='*60}")

    _run([python, "data_pipeline/build_knowledge_base.py", "--corpus", corpus])
    _run([python, "data_pipeline/build_vector_index.py", "--corpus", corpus])

    print(f"\n{'='*60}")
    print(f"  ✅  Ingest complete for corpus: {corpus!r}")
    print(f"{'='*60}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def check_corpus(corpus: str | None = None, *, ingest: bool = False, force: bool = False) -> dict:
    """
    Check corpus freshness and optionally re-ingest.

    Returns a dict with keys: corpus, up_to_date, new_files, changed_files,
    removed_files, unparsed_pdfs, ingested.
    """
    resolved = get_corpus_name(corpus)

    current_texts = _scan_parsed_texts(resolved)
    unparsed_pdfs = _scan_raw_pdfs(resolved)
    fingerprint = _load_fingerprint(resolved)
    known_texts: dict = fingerprint.get("sources", {})

    diff = _diff(current_texts, known_texts)
    _print_report(resolved, diff, unparsed_pdfs)

    ingested = False
    needs_ingest = (not diff["up_to_date"]) or unparsed_pdfs

    if ingest and (needs_ingest or force):
        _ingest(resolved)
        # Refresh scan after pipeline completes (PDF parser may have added new .txt files)
        current_texts = _scan_parsed_texts(resolved)
        unparsed_pdfs = _scan_raw_pdfs(resolved)
        _save_fingerprint(resolved, current_texts, unparsed_pdfs)
        print(f"  💾  Fingerprint saved → {_fingerprint_path(resolved)}")
        ingested = True
    elif ingest and not needs_ingest and not force:
        print("  ℹ️   Nothing to ingest (use --force to rebuild anyway).\n")
    elif not ingest and needs_ingest:
        print("  ℹ️   Tip: re-run with --ingest to update the knowledge base.\n")

    # First-time fingerprint creation (report-only run on a fresh corpus)
    if not _fingerprint_path(resolved).is_file() and not ingest:
        _save_fingerprint(resolved, current_texts, unparsed_pdfs)
        print(f"  💾  Fingerprint initialised → {_fingerprint_path(resolved)}\n")

    return {
        "corpus": resolved,
        **diff,
        "unparsed_pdfs": unparsed_pdfs,
        "ingested": ingested,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check corpus freshness and optionally re-run the ingestion pipeline.",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Corpus to check (defaults to CORPUS_NAME env or 'public').",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run the full pipeline and rebuild the vector index if updates are detected.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a full rebuild even when the fingerprint reports no changes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = check_corpus(corpus=args.corpus, ingest=args.ingest, force=args.force)
    sys.exit(0 if result["up_to_date"] or result["ingested"] else 1)
