"""
PDF parser - step 1 of the offline ingestion pipeline.

For every PDF in the active corpus's ``raw/`` directory this module produces
two artefacts in the corpus's ``parsed_text/`` directory:
    * ``{slug}.txt``       - the full extracted text (joined page-by-page)
    * ``{slug}.meta.json`` - a metadata sidecar with author, title, year, page
                             count, character count, and parser notes

The parser is corpus-aware via :func:`shared_components.utilities.path_utils`
so the same code works for the public dossier corpus and for the private
full-book corpus.

Hardening for full-length books:
    * Joins lines that were soft-broken with hyphenation ("commit-\\nment").
    * Collapses internal whitespace runs introduced by multi-column layouts.
    * Looks for richer metadata patterns in the first few pages.
    * Records ``parser_notes`` whenever a field could not be inferred so the
      gap is visible downstream.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz  # PyMuPDF - high-fidelity PDF text extraction.

try:
    from data_pipeline.bootstrap import add_project_root_to_sys_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_pipeline.bootstrap import add_project_root_to_sys_path

add_project_root_to_sys_path()

from shared_components.utilities.path_utils import (
    ensure_directory,
    get_parsed_text_dir,
    get_raw_data_dir,
)


# Regex helpers used during text cleanup.
_HYPHEN_LINEBREAK = re.compile(r"-\n(?=[a-z])")
_RUNS_OF_BLANK_LINES = re.compile(r"\n{3,}")


def infer_title_from_filename(file_name: str) -> str:
    """Derive a human-readable title from a PDF filename.

    Strips trailing numeric ids (e.g. ISBNs) and hyphens, then title-cases the
    remainder. Used as a deterministic fallback when no explicit title can be
    extracted from the document body.
    """
    base_name = Path(file_name).stem
    cleaned = re.sub(r"-\d{10,}$", "", base_name)
    cleaned = cleaned.replace("-", " ").strip()
    return cleaned.title()


def normalise_extracted_text(raw_text: str) -> str:
    """Repair common artefacts of PDF text extraction.

    PyMuPDF preserves visual line breaks even when a word was hyphenated
    across them. We rejoin those breaks ("commit-\\nment" -> "commitment")
    and collapse triple-or-more newlines down to two so paragraph structure
    survives without the page-break noise. Single newlines are preserved so
    chapter detection later in the pipeline can still locate them.
    """
    text = _HYPHEN_LINEBREAK.sub("", raw_text)
    text = _RUNS_OF_BLANK_LINES.sub("\n\n", text)
    return text


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, int]:
    """Extract the full text body and page count from a PDF.

    Pages are joined with double newlines so downstream chunk- and chapter-
    detection logic can find page boundaries when needed. The return value is
    ``(normalised_text, page_count)``.
    """
    with fitz.open(pdf_path) as document:
        pages = [page.get_text() for page in document]
        page_count = len(document)

    joined = "\n\n".join(pages)
    return normalise_extracted_text(joined), page_count


def extract_metadata_from_first_pages(pdf_path: Path, file_name: str, max_pages: int = 6) -> dict:
    """Pull author and publication-year metadata from the opening pages.

    The first few pages of nearly every book carry copyright, authorship,
    title, and publication date - they are the most reliable source of
    metadata short of a parsed PDF metadata stream (which is often missing
    or wrong).

    Strategy:
        * Use a series of regexes that match common authorship conventions
          ("Authors:", "Copyright (c) YEAR by NAME", "by NAME").
        * Look for a publication year using the same kinds of patterns.
        * Fall back to ``infer_title_from_filename`` for the title and check
          for a "Title: ..." override on the first pages.
        * Record any field that could not be resolved into ``parser_notes``
          so the gap is visible in the sidecar instead of silently lost.
    """
    with fitz.open(pdf_path) as document:
        first_pages_text = [document[i].get_text() for i in range(min(max_pages, len(document)))]

    first_text = normalise_extracted_text("\n\n".join(first_pages_text))
    parser_notes: list[str] = []

    # Order matters: more specific patterns first so a generic "by X" match
    # does not steal from a stronger explicit "Authors:" line.
    author_patterns = [
        r"Authors?:\s+(.+?)(?:\n|Publication context|Corpus priority|Document type)",
        r"Copyright\s+\u00a9\s+\d{4}\s+by\s+(.+?)(?:\n|Cover|Contents|ISBN)",
        r"by\s+([A-Z][^\n]{2,120})(?:\n|Contents|ISBN)",
    ]

    author = "unknown"
    for pattern in author_patterns:
        match = re.search(pattern, first_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            author = re.sub(r"\s+", " ", match.group(1)).strip()
            break

    if author == "unknown":
        parser_notes.append("author_not_found_in_first_pages")

    # Publication year heuristics: explicit "First edition: <month> <year>" or
    # the first copyright year on the page.
    year_match = re.search(
        r"(?:First .{1,40} edition[:,]\s+\w+\s+|Copyright\s+\u00a9\s+|First published in\s+)(\d{4})",
        first_text,
        flags=re.IGNORECASE,
    )
    publication_year = int(year_match.group(1)) if year_match else None

    if publication_year is None:
        parser_notes.append("publication_year_not_found_in_first_pages")

    # Allow an explicit "Title: ..." override; otherwise fall back to filename inference.
    title_match = re.search(r"Title:\s+([^\n]{3,160})", first_text)
    title = title_match.group(1).strip() if title_match else infer_title_from_filename(file_name)

    return {
        "author": author,
        "title": title,
        "publication_year": publication_year,
        "parser_notes": parser_notes,
    }


def parse_all_pdfs(corpus: str | None = None):
    """Parse every PDF in the active corpus's raw directory.

    Args:
        corpus: Optional corpus name override. When omitted, the value of the
            ``CORPUS_NAME`` environment variable wins, falling back to ``"public"``.

    Returns:
        A list of per-PDF result records (status ``ok`` / ``failed``) suitable
        for an orchestrator to log a summary without re-reading the output dir.
    """

    input_dir = get_raw_data_dir(corpus)
    output_dir = ensure_directory(get_parsed_text_dir(corpus))

    results = []
    print(f"\nReading PDFs from:\n{input_dir}\n")

    if not input_dir.exists():
        print(f"  (no raw directory at {input_dir} - nothing to parse)")
        return results

    for pdf_path in sorted(input_dir.glob("*.pdf")):
        print(f"Parsing: {pdf_path.name}")
        source_name = pdf_path.stem

        try:
            text, page_count = extract_text_from_pdf(pdf_path)
            metadata = extract_metadata_from_first_pages(pdf_path, pdf_path.name)
            metadata.update(
                {
                    "file_name": pdf_path.name,
                    "source": source_name,
                    "page_count": page_count,
                    "character_count": len(text),
                }
            )
        except Exception as exc:  # pragma: no cover - parser failure path
            # One bad PDF should not fail the whole batch; record the failure
            # and move on so the rest of the corpus still ingests.
            print(f"  Failed to parse {pdf_path.name}: {exc}")
            results.append(
                {
                    "source": source_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            continue

        text_output_file = output_dir / f"{source_name}.txt"
        metadata_output_file = output_dir / f"{source_name}.meta.json"

        text_output_file.write_text(text, encoding="utf-8")
        metadata_output_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        results.append({"metadata": metadata, "text_file": str(text_output_file), "status": "ok"})
        print(f"  Author: {metadata['author']}")
        print(f"  Title: {metadata['title']}")
        print(f"  Pages: {page_count} | Characters: {len(text):,}")

    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. ``--corpus`` overrides the env-driven default."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        default=None,
        help="Corpus name to parse (e.g. 'public', 'private'). Defaults to CORPUS_NAME env or 'public'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    parse_all_pdfs(corpus=args.corpus)
