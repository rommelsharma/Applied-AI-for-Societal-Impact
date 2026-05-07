"""
Create traceable semantic chunks from parsed source documents.
"""

from __future__ import annotations

import json
import re

from pathlib import Path

try:
    from data_pipeline.bootstrap import add_project_root_to_sys_path
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data_pipeline.bootstrap import add_project_root_to_sys_path

add_project_root_to_sys_path()

from shared_components.utilities.path_utils import (
    ensure_directory,
    get_chunks_dir,
    get_parsed_text_dir,
)

INPUT_DIR = get_parsed_text_dir()
OUTPUT_DIR = ensure_directory(get_chunks_dir())
OUTPUT_FILE = OUTPUT_DIR / "chunks.json"

CHUNK_SIZE = 350
CHUNK_OVERLAP = 60

FRONT_MATTER_MARKERS = [
    "introduction",
    "part i",
    "chapter 1",
]

BACK_MATTER_MARKERS = [
    "acknowledgments",
    "notes",
    "discover more",
    "about the authors",
    "also by ",
]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def trim_document_text(raw_text: str) -> tuple[str, list[str]]:
    lowered = raw_text.lower()
    notes: list[str] = []
    start_index = 0
    end_index = len(raw_text)

    front_search_floor = max(len(raw_text) // 20, 1)
    front_hits = [
        lowered.find(marker, front_search_floor)
        for marker in FRONT_MATTER_MARKERS
        if lowered.find(marker, front_search_floor) >= 0
    ]
    if front_hits:
        start_index = min(front_hits)
        notes.append("front_matter_trimmed")

    back_search_floor = max(start_index + 1, len(raw_text) // 2)
    back_hits = [
        lowered.find(marker, back_search_floor)
        for marker in BACK_MATTER_MARKERS
        if lowered.find(marker, back_search_floor) >= 0
    ]
    if back_hits:
        end_index = min(back_hits)
        notes.append("back_matter_trimmed")

    trimmed = raw_text[start_index:end_index]
    return trimmed, notes


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def load_metadata(meta_file_path):
    with open(meta_file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def create_chunks():
    all_chunks = []
    print(f"\nReading parsed text from:\n{INPUT_DIR}\n")

    for text_path in sorted(INPUT_DIR.glob("*.txt")):
        source_name = text_path.stem
        metadata_path = INPUT_DIR / f"{source_name}.meta.json"
        print(f"Chunking: {source_name}")

        raw_text = text_path.read_text(encoding="utf-8")
        trimmed_text, cleaning_notes = trim_document_text(raw_text)
        cleaned_text = clean_text(trimmed_text)
        metadata = load_metadata(metadata_path)
        chunks = chunk_text(cleaned_text)

        print(f"  Chunks created: {len(chunks)}")

        for idx, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "id": f"{source_name}_chunk_{idx}",
                    "source": metadata["source"],
                    "file_name": metadata["file_name"],
                    "author": metadata["author"],
                    "title": metadata.get("title"),
                    "publication_year": metadata.get("publication_year"),
                    "parser_notes": metadata.get("parser_notes", []),
                    "cleaning_notes": cleaning_notes,
                    "chunk_index": idx,
                    "word_count": len(chunk.split()),
                    "text": chunk,
                }
            )

    print(f"\nSaving chunks to:\n{OUTPUT_FILE}")
    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(all_chunks, handle, indent=2)

    print(f"\n✅ Total chunks saved: {len(all_chunks)}")


if __name__ == "__main__":
    create_chunks()
