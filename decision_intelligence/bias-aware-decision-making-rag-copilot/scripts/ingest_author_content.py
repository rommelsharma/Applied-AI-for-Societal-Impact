"""
Phase 2 ingestion: pull author-published content from the open web (or from
already-downloaded local files) into the public corpus.

Reads ``data/metadata/author_content_manifest.json`` and, for each entry,
writes a normalised plain-text file plus a ``.meta.json`` sidecar into the
target corpus's ``parsed_text`` directory. After this script completes, the
standard pipeline takes over::

    python data_pipeline/build_knowledge_base.py --corpus public --skip-parse
    python data_pipeline/build_vector_index.py --corpus public

Design choices:
    * Uses only the Python stdlib so no new dependency is required.
    * Strips HTML using ``html.parser`` (no third-party library needed).
    * Refuses to ingest any item whose ``license_status`` is
      ``review_required``; the operator must explicitly confirm licensing
      before content joins the public corpus.
    * Honours a polite rate limit between fetches and identifies itself with
      a UA string so site operators know who is calling.

This is the operating tool for Phase 2 of ``docs/phased-rollout-plan.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from shared_components.utilities.path_utils import (
    ensure_directory,
    get_corpus_name,
    get_parsed_text_dir,
)


DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "metadata" / "author_content_manifest.json"
DEFAULT_USER_AGENT = (
    "bias-aware-decision-rag-copilot/0.1 "
    "(+https://github.com/rommelsharma/Applied-AI-for-Societal-Impact)"
)
DEFAULT_RATE_LIMIT_SECONDS = 1.0
DEFAULT_FETCH_TIMEOUT_SECONDS = 30


# Tags whose textual content should be discarded outright when stripping HTML.
_DROP_TAGS = {
    "script", "style", "noscript", "nav", "header", "footer",
    "aside", "form", "iframe", "svg", "button", "menu", "title",
}
_BLOCK_TAGS = {
    "p", "br", "hr", "div", "section", "article", "li", "ul", "ol",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "tr", "td", "th",
}


class _TextExtractor(HTMLParser):
    """Convert HTML into readable plain text.

    Skips content inside tags listed in ``_DROP_TAGS`` (navigation, scripts,
    etc.) and inserts a newline whenever a block-level boundary is crossed
    so paragraphs survive the conversion.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._buffer: list[str] = []
        self._suppress_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _DROP_TAGS:
            self._suppress_depth += 1
        elif tag in _BLOCK_TAGS:
            self._buffer.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _DROP_TAGS and self._suppress_depth > 0:
            self._suppress_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._buffer.append("\n")

    def handle_data(self, data: str) -> None:
        if self._suppress_depth == 0 and data:
            self._buffer.append(data)

    def text(self) -> str:
        return "".join(self._buffer)


def _strip_html(html: str) -> str:
    """Convert HTML to clean plain text with paragraph breaks preserved."""
    extractor = _TextExtractor()
    extractor.feed(html)
    text = extractor.text()
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _is_html(url_or_path: str, body: str) -> bool:
    """Heuristic: treat as HTML if URL ends in html/htm or body opens with a tag."""
    lower = url_or_path.lower()
    if lower.endswith((".html", ".htm")):
        return True
    if lower.endswith((".txt", ".md")):
        return False
    sniff = body.lstrip()[:200].lower()
    return sniff.startswith("<!doctype html") or sniff.startswith("<html")


def _fetch_url(url: str, *, user_agent: str, timeout: int) -> str:
    """Fetch ``url`` and return the response body as a string."""
    request = Request(url, headers={"User-Agent": user_agent, "Accept": "*/*"})
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _read_local(path: Path) -> str:
    """Read a local file as text, normalising line endings."""
    return path.read_text(encoding="utf-8", errors="replace")


def _validate_item(item: dict, *, slug: str) -> None:
    """Ensure the manifest entry has the fields we will rely on downstream."""
    required = {"slug", "title", "author", "content_type", "license_status"}
    missing = required - set(item)
    if missing:
        raise ValueError(f"Manifest entry '{slug}' is missing fields: {sorted(missing)}")

    if not item.get("source_url") and not item.get("local_path"):
        raise ValueError(
            f"Manifest entry '{slug}' must declare either source_url or local_path"
        )

    status = item["license_status"]
    if status == "review_required":
        raise ValueError(
            f"Manifest entry '{slug}' has license_status='review_required'. "
            "Confirm licensing and update the manifest before ingestion."
        )


def _build_meta_json(item: dict, *, char_count: int, fetch_origin: str) -> dict:
    """Construct the ``.meta.json`` payload that mirrors PDF parser output."""
    parser_notes = ["author_content_ingest", f"fetched_from:{fetch_origin}"]
    if not item.get("publication_year"):
        parser_notes.append("publication_year_not_provided")

    return {
        "author": item["author"],
        "title": item["title"],
        "publication_year": item.get("publication_year"),
        "parser_notes": parser_notes,
        "file_name": f"{item['slug']}.{ 'html' if fetch_origin == 'url' else 'src' }",
        "source": item["slug"],
        "page_count": None,
        "character_count": char_count,
        "publisher": item.get("publisher"),
        "content_type": item["content_type"],
        "source_url": item.get("source_url"),
        "license_status": item["license_status"],
        "license_notes": item.get("license_notes"),
        "tags": item.get("tags", []),
        "decision_domains": item.get("decision_domains", []),
    }


def ingest_manifest(
    *,
    manifest_path: Path,
    corpus: str,
    only_slug: str | None,
    dry_run: bool,
    user_agent: str,
    rate_limit_seconds: float,
    fetch_timeout: int,
) -> int:
    """Drive the full ingestion run; returns the count of items written.

    Skips items already present unless ``--force`` is passed (not implemented
    here intentionally - the operator deletes the file to re-ingest, which
    keeps the audit trail simple).
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    if not items:
        print(f"[ingest] manifest is empty - nothing to do ({manifest_path})")
        return 0

    output_dir = ensure_directory(get_parsed_text_dir(corpus))
    print(
        f"[ingest] manifest='{manifest_path}' corpus='{corpus}' items={len(items)} "
        f"target='{output_dir}' dry_run={dry_run}"
    )

    written = 0
    for index, item in enumerate(items):
        slug = item.get("slug", f"item_{index}")
        if only_slug and slug != only_slug:
            continue
        try:
            _validate_item(item, slug=slug)
        except ValueError as exc:
            print(f"[skip] {slug}: {exc}")
            continue

        text_target = output_dir / f"{slug}.txt"
        meta_target = output_dir / f"{slug}.meta.json"
        if text_target.exists():
            print(f"[skip] {slug}: already ingested at {text_target} - delete to re-ingest")
            continue

        try:
            if item.get("source_url"):
                origin = "url"
                if dry_run:
                    print(f"[plan] {slug}: would fetch {item['source_url']}")
                    continue
                body = _fetch_url(
                    item["source_url"], user_agent=user_agent, timeout=fetch_timeout
                )
                time.sleep(rate_limit_seconds)
            else:
                origin = "local"
                local = Path(item["local_path"])
                if not local.is_absolute():
                    local = PROJECT_ROOT / local
                if dry_run:
                    print(f"[plan] {slug}: would read {local}")
                    continue
                body = _read_local(local)

            if _is_html(item.get("source_url") or item.get("local_path", ""), body):
                text = _strip_html(body)
            else:
                text = body.strip() + "\n"

            text_target.write_text(text, encoding="utf-8")
            meta = _build_meta_json(item, char_count=len(text), fetch_origin=origin)
            meta_target.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            written += 1
            print(f"[ok]   {slug}: {len(text):,} chars -> {text_target.name}")

        except (HTTPError, URLError) as net_err:
            print(f"[fail] {slug}: network error - {net_err}")
        except FileNotFoundError as fs_err:
            print(f"[fail] {slug}: local file missing - {fs_err}")
        except Exception as unexpected:  # pragma: no cover - defensive
            print(f"[fail] {slug}: {type(unexpected).__name__}: {unexpected}")

    print(f"[ingest] complete - {written} item(s) written")
    return written


def parse_args() -> argparse.Namespace:
    """Configure the CLI surface for ``ingest_author_content.py``."""
    parser = argparse.ArgumentParser(
        description="Ingest author-published web content into the public corpus.",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Corpus name (default: $CORPUS_NAME or 'public').",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to the author content manifest JSON file.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Limit ingestion to a single manifest item (by slug).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned actions without writing or fetching.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent string for outbound HTTP requests.",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=DEFAULT_RATE_LIMIT_SECONDS,
        help="Seconds to wait between successive fetches (politeness).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_FETCH_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    corpus = args.corpus or get_corpus_name()
    ingest_manifest(
        manifest_path=Path(args.manifest),
        corpus=corpus,
        only_slug=args.only,
        dry_run=args.dry_run,
        user_agent=args.user_agent,
        rate_limit_seconds=args.rate_limit,
        fetch_timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
