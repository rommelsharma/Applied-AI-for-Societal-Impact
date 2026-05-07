"""
Parse PDF source documents into text and metadata sidecars.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # PyMuPDF

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

INPUT_DIR = get_raw_data_dir()
OUTPUT_DIR = ensure_directory(get_parsed_text_dir())


def infer_title_from_filename(file_name: str) -> str:
    base_name = Path(file_name).stem
    cleaned = re.sub(r"-\d{10,}$", "", base_name)
    cleaned = cleaned.replace("-", " ").strip()
    return cleaned.title()


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, int]:
    with fitz.open(pdf_path) as document:
        text = [page.get_text() for page in document]
        return "\n".join(text), len(document)


def extract_metadata_from_first_pages(pdf_path: Path, file_name: str, max_pages: int = 5) -> dict:
    with fitz.open(pdf_path) as document:
        first_pages_text = [document[i].get_text() for i in range(min(max_pages, len(document)))]

    first_text = "\n".join(first_pages_text)
    parser_notes: list[str] = []

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

    year_match = re.search(r"(?:First .* edition:\s+\w+\s+|Copyright\s+\u00a9\s+)(\d{4})", first_text)
    publication_year = int(year_match.group(1)) if year_match else None

    if publication_year is None:
        parser_notes.append("publication_year_not_found_in_first_pages")

    title = infer_title_from_filename(file_name)

    return {
        "author": author,
        "title": title,
        "publication_year": publication_year,
        "parser_notes": parser_notes,
    }


def parse_all_pdfs():
    """Parse all PDFs in the raw data directory."""

    results = []
    print(f"\nReading PDFs from:\n{INPUT_DIR}\n")

    for pdf_path in sorted(INPUT_DIR.glob("*.pdf")):
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
            print(f"  Failed to parse {pdf_path.name}: {exc}")
            results.append(
                {
                    "source": source_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            continue

        text_output_file = OUTPUT_DIR / f"{source_name}.txt"
        metadata_output_file = OUTPUT_DIR / f"{source_name}.meta.json"

        text_output_file.write_text(text, encoding="utf-8")
        metadata_output_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        results.append({"metadata": metadata, "text_file": str(text_output_file), "status": "ok"})
        print(f"  Author: {metadata['author']}")
        print(f"  Title: {metadata['title']}")

    return results


if __name__ == "__main__":
    parse_all_pdfs()
