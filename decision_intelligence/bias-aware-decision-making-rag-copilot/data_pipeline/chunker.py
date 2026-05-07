"""
Chunker - step 2 of the offline ingestion pipeline.

Reads each parsed text plus its metadata sidecar, trims known front- and
back-matter, normalises whitespace, detects chapter boundaries, and splits
the body into overlapping word-windowed chunks suitable for embedding and
retrieval.

Profile-driven sizing:
    * ``dossier`` - 350 words / 60 overlap. Suited to short syntheses.
    * ``book``    - 500 words / 80 overlap. Suited to full-length books.
    * ``auto``    - decide per-document by length: > 30,000 words uses
                    ``book``, otherwise ``dossier``.

Why these defaults:
    * 350 words is roughly the length of a single concept-dense argument in
      decision-science writing - long enough to embed meaningfully, short
      enough to retrieve precisely.
    * 500 words gives full books extra headroom so a worked example or a
      named study fits inside one chunk together with its surrounding
      framing.
    * 60-80 word overlap keeps key sentences from being split across chunk
      boundaries so the retrieved chunk usually contains the full thought.
"""

from __future__ import annotations

import argparse
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

from shared_components.settings import RAG_SETTINGS
from shared_components.utilities.path_utils import (
    ensure_directory,
    get_chunks_dir,
    get_parsed_text_dir,
)


# Chunking profiles. ``auto`` resolves to one of the named profiles per-document.
PROFILES = {
    "dossier": {"chunk_size": 350, "overlap": 60},
    "book": {"chunk_size": 500, "overlap": 80},
}
AUTO_BOOK_THRESHOLD_WORDS = 30_000

# Section-heading patterns used to bound real content. We use line-anchored,
# end-of-line-terminated regex matching rather than substring search because
# words like "notes", "references", and "index" appear as inline citations
# throughout the body of academic-style books and would otherwise misfire.
#
# A negative lookahead excludes table-of-contents lines, where the heading
# is followed by dot leaders and a page number ("Chapter 1 ........ 5").

_FRONT_HEADING = re.compile(
    r"^[ \t]*(?:"
    r"CHAPTER\s+(?:1|ONE)\b|Chapter\s+(?:1|One)\b|"
    r"PART\s+(?:I|ONE|1)\b|Part\s+(?:I|One|1)\b|"
    r"INTRODUCTION\b|Introduction\b|"
    r"PROLOGUE\b|Prologue\b|"
    r"PREFACE\b|Preface\b|"
    r"FOREWORD\b|Foreword\b"
    r")(?!\s*\.{2,}|\s+\d{1,4}\s*$)",
    flags=re.MULTILINE,
)

# Back-matter heading patterns are intentionally strict: most require the
# heading to terminate the line so an inline body reference ("see notes
# below") cannot trigger a trim. Only headings that have a clear textual
# tail (e.g. "Acknowledgments by Daniel Kahneman") are slightly relaxed.
_BACK_HEADING = re.compile(
    r"^[ \t]*(?:"
    r"NOTES\s*$|Notes\s*$|"
    r"ENDNOTES\s*$|Endnotes\s*$|"
    r"ACKNOWLEDG(?:E?)MENTS\b|Acknowledg(?:e?)ments\b|"
    r"BIBLIOGRAPHY\s*$|Bibliography\s*$|"
    r"REFERENCES\s*$|References\s*$|"
    r"INDEX\s*$|Index\s*$|"
    r"GLOSSARY\s*$|Glossary\s*$|"
    r"APPENDIX(?:\s+[A-Z][A-Z0-9]?)?\s*$|Appendix(?:\s+[A-Z][A-Za-z0-9]?)?\s*$|"
    r"ABOUT\s+THE\s+AUTHOR[S]?\b|About\s+the\s+Author[s]?\b|"
    r"PERMISSIONS\s*$|Permissions\s*$|"
    r"COPYRIGHT\s+ACKNOWLEDG\w*|Copyright\s+Acknowledg\w*|"
    r"DISCOVER\s+MORE\b|Discover\s+More\b|"
    r"Also\s+by\s+\w|ALSO\s+BY\s+\w"
    r")",
    flags=re.MULTILINE,
)

# Backwards-compatible aliases retained for any external caller / test that
# imported the marker lists by name.
FRONT_MATTER_MARKERS = [
    "introduction", "preface", "foreword", "prologue",
    "part i", "part one", "chapter 1", "chapter one",
]
BACK_MATTER_MARKERS = [
    "acknowledgments", "acknowledgements", "notes", "endnotes",
    "bibliography", "references", "index", "glossary", "appendix",
    "discover more", "about the authors", "about the author",
    "also by ", "permissions", "copyright acknowledgments",
]

# Front matter is searched in the first 60% of the document (covers the
# unusual case of TFAS-style PDFs where an index sits at the front);
# back matter is searched only in the last 25% to keep inline body
# references from triggering an early trim.
_FRONT_SEARCH_FRACTION = 0.60
_BACK_SEARCH_FRACTION = 0.75
# If trimming would discard more than 70% of the document, the heuristics
# probably misfired - keep the document intact and emit a flag instead.
_TRIM_SAFETY_THRESHOLD = 0.30

# Chapter detection: matches "Chapter 12", "Chapter Twelve", "CHAPTER 3 -",
# and "12. Title On A Line" near the start of a line. Captures the title
# portion (if present on the same line) into group 1 - falsy when absent.
_CHAPTER_HEADING = re.compile(
    r"^\s*(?:chapter\s+(?:[0-9]+|[a-z]+)|\d{1,3}\s*[\.:\-])\s*([A-Z][^\n]{0,120})?\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def clean_text(text: str) -> str:
    """Collapse all whitespace runs into single spaces.

    PDF extraction often leaves stray newlines and double spaces; the
    embedder treats these as noise, so a normalised single-space form yields
    cleaner embeddings without altering meaning.
    """
    return re.sub(r"\s+", " ", text).strip()


def trim_document_text(raw_text: str) -> tuple[str, list[str]]:
    """Trim front matter and back matter from a parsed document body.

    Uses line-anchored regex heading detection rather than naive substring
    search so words like "notes", "index", and "references" used as inline
    citations in the body cannot trigger an early back-matter trim.

    Behaviour:
        * Front matter: the FIRST clean heading match in the first 60% of
          the document marks the start of the body. The 60% bound covers
          unusual PDFs whose front matter, table of contents, or a stray
          index occupies a large fraction of the file.
        * Back matter: the EARLIEST clean heading match in the last 25% of
          the document marks the start of back matter. The end-of-line
          anchor on most patterns prevents inline body references from
          firing.
        * Safety net: if the proposed trim would discard more than 70% of
          the document, it is rejected as a likely heuristic misfire and
          the original text is returned with a ``trim_skipped_too_aggressive``
          note so reviewers can spot the case.

    Returns the trimmed text plus a list of cleaning notes describing what
    was trimmed - those notes are persisted on every chunk for auditability.
    """
    notes: list[str] = []
    total_len = len(raw_text)
    if total_len < 1000:
        # Document too short for the heuristics to be safe; pass through.
        return raw_text, notes

    start_index = 0
    end_index = total_len

    front_window_end = max(int(total_len * _FRONT_SEARCH_FRACTION), 1)
    front_match = _FRONT_HEADING.search(raw_text, 0, front_window_end)
    if front_match:
        start_index = front_match.start()
        notes.append("front_matter_trimmed")

    back_window_start = max(start_index + 1, int(total_len * _BACK_SEARCH_FRACTION))
    back_match = _BACK_HEADING.search(raw_text, back_window_start)
    if back_match:
        end_index = back_match.start()
        notes.append("back_matter_trimmed")

    trimmed = raw_text[start_index:end_index]
    if len(trimmed) < total_len * _TRIM_SAFETY_THRESHOLD:
        # The heuristic almost certainly mis-fired. Keep the original text
        # so the chunker can still produce a usable knowledge base.
        return raw_text, [*notes, "trim_skipped_too_aggressive"]

    return trimmed, notes


def detect_chapters(text: str) -> list[tuple[int, str]]:
    """Locate chapter boundaries in the (still-newline-preserving) text.

    Returns a sorted list of ``(start_offset, chapter_title)`` tuples - one
    entry per detected chapter. ``chapter_title`` is best-effort: for "Chapter
    Twelve" without a title on the same line it is just "Chapter Twelve".

    Whitespace in the captured heading is normalised so a heading split across
    two lines in the source PDF (``CHAPTER 7\\nOccasion Noise``) renders as a
    clean single-line title (``CHAPTER 7 Occasion Noise``).
    """
    boundaries: list[tuple[int, str]] = []
    for match in _CHAPTER_HEADING.finditer(text):
        # match.group(0) can include unicode line separators that the regex
        # didn't strictly treat as newlines; collapse them so chapter_title
        # is always a single tidy line.
        heading = re.sub(r"\s+", " ", match.group(0)).strip()
        boundaries.append((match.start(), heading))
    return boundaries


def assign_chapter_titles(text: str, total_chunk_words: int, chunk_word_starts: list[int]) -> list[str]:
    """Map each chunk's starting word index to the most recent chapter title.

    The chunker operates on whitespace-collapsed text, but chapter detection
    happens on the version that still has newlines. To bridge the two we
    match the cumulative *word offset* of each chapter heading against the
    word offset where each chunk starts, and assign the most recent heading.

    Returns a list aligned 1:1 with the chunks.
    """
    boundaries = detect_chapters(text)
    if not boundaries:
        return [""] * len(chunk_word_starts)

    # Translate character offsets to word indices on the same trimmed text so
    # the comparison with chunk_word_starts is consistent.
    boundary_word_indices: list[tuple[int, str]] = []
    for char_offset, title in boundaries:
        prefix = text[:char_offset]
        word_index = len(prefix.split())
        boundary_word_indices.append((word_index, title))

    boundary_word_indices.sort(key=lambda pair: pair[0])

    titles: list[str] = []
    for chunk_start in chunk_word_starts:
        current_title = ""
        for word_index, title in boundary_word_indices:
            if word_index <= chunk_start:
                current_title = title
            else:
                break
        titles.append(current_title)

    return titles


def select_profile(text: str, profile: str | None = None) -> tuple[str, dict]:
    """Resolve the chunking profile to use and return its parameters.

    Profile precedence: explicit argument -> env-configured -> auto-detect.
    """
    chosen = profile or RAG_SETTINGS.chunk_profile or "auto"

    if chosen == "auto":
        word_count = len(text.split())
        chosen = "book" if word_count >= AUTO_BOOK_THRESHOLD_WORDS else "dossier"

    if chosen not in PROFILES:
        raise ValueError(f"Unknown chunk profile: {chosen!r} - expected one of {list(PROFILES)}")

    return chosen, PROFILES[chosen]


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> tuple[list[str], list[int]]:
    """Split text into overlapping word-windowed chunks.

    Word-level (rather than character-level) chunking keeps boundaries on
    whole tokens, which matters because the embedding model tokenises by
    sub-word units; clean word boundaries reduce noise at chunk edges.

    Returns ``(chunks, starting_word_indices)`` so callers can correlate each
    chunk back to its position in the original text (used for chapter-title
    assignment).
    """
    words = text.split()
    chunks: list[str] = []
    starts: list[int] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
            starts.append(start)
        start += chunk_size - overlap

    return chunks, starts


def load_metadata(meta_file_path):
    """Load the metadata sidecar JSON written by the PDF parser."""
    with open(meta_file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def create_chunks(corpus: str | None = None, profile: str | None = None):
    """Build chunks for every parsed document and persist them as a single JSON file.

    Each chunk inherits source metadata (author, title, year, file_name) and
    additionally carries:
        * ``id``                   - stable identifier ``{slug}_chunk_{index}``
        * ``chunk_index``          - position within its source document
        * ``word_count``           - audit field
        * ``parser_notes`` /
          ``cleaning_notes``       - traceability flags
        * ``chapter_title``        - the chapter the chunk falls inside (if detectable)
        * ``chunking_profile``     - "dossier" / "book" - used to evaluate retrieval lift

    Args:
        corpus: Optional corpus name override (defaults to ``CORPUS_NAME`` env).
        profile: Optional chunking-profile override; ``None`` defers to env or auto-detect.
    """
    input_dir = get_parsed_text_dir(corpus)
    output_dir = ensure_directory(get_chunks_dir(corpus))
    output_file = output_dir / "chunks.json"

    all_chunks = []
    print(f"\nReading parsed text from:\n{input_dir}\n")

    for text_path in sorted(input_dir.glob("*.txt")):
        source_name = text_path.stem
        metadata_path = input_dir / f"{source_name}.meta.json"
        print(f"Chunking: {source_name}")

        raw_text = text_path.read_text(encoding="utf-8")
        trimmed_text, cleaning_notes = trim_document_text(raw_text)

        chosen_profile, params = select_profile(trimmed_text, profile)
        cleaned_text = clean_text(trimmed_text)
        metadata = load_metadata(metadata_path)
        chunks, starts = chunk_text(
            cleaned_text,
            chunk_size=params["chunk_size"],
            overlap=params["overlap"],
        )

        # Assign chapter titles using the still-newline-preserving trimmed text
        # so chapter heading detection has line-anchored regex to work with.
        chapter_titles = assign_chapter_titles(trimmed_text, len(cleaned_text.split()), starts)

        print(f"  Profile: {chosen_profile} | Chunks created: {len(chunks)}")

        for idx, (chunk, chapter_title) in enumerate(zip(chunks, chapter_titles)):
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
                    "chunking_profile": chosen_profile,
                    "chapter_title": chapter_title,
                    "chunk_index": idx,
                    "word_count": len(chunk.split()),
                    "text": chunk,
                }
            )

    print(f"\nSaving chunks to:\n{output_file}")
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(all_chunks, handle, indent=2)

    print(f"\n✅ Total chunks saved: {len(all_chunks)}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. Both ``--corpus`` and ``--profile`` are optional."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=None, help="Corpus to chunk (defaults to CORPUS_NAME or 'public').")
    parser.add_argument(
        "--profile",
        default=None,
        choices=["dossier", "book", "auto"],
        help="Chunking profile override.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_chunks(corpus=args.corpus, profile=args.profile)
