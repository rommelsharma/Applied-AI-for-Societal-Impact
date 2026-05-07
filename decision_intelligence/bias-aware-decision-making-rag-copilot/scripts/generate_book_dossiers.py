from __future__ import annotations

import json
import html
import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOOKS_FILE = ROOT / "data" / "metadata" / "book_dossiers" / "phase1_priority_books.json"
OUTPUT_MD_DIR = ROOT / "data" / "metadata" / "book_dossiers" / "generated_markdown"
OUTPUT_HTML_DIR = ROOT / "data" / "metadata" / "book_dossiers" / "generated_html"
OUTPUT_PDF_DIR = ROOT / "data" / "raw"


def bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_markdown(book: dict) -> str:
    title = book["title"]
    authors = ", ".join(book["authors"])
    sources = "\n".join(
        f"- {source['label']} ({source['type']}): {source['url']}"
        for source in book["sources"]
    )

    chapter_section = ""
    chapter_summaries = book.get("chapter_summaries", [])
    if chapter_summaries:
        chapter_lines = "\n".join(
            f"- **{chapter['title']}**: {chapter['summary']}"
            for chapter in chapter_summaries
        )
        chapter_section = f"""
## Chapter-Wise Summary
{chapter_lines}
"""

    return f"""# {title}

**Authors:** {authors}  
**Publication context:** {book["year"]}  
**Corpus priority:** {book["priority_phase"]}  
**Document type:** Synthesized RAG source dossier built from publicly available online material

## Purpose In This RAG Corpus
{bullet_list(book["why_important"])}

## Key Concepts And Retrieval Tags
{bullet_list(book["key_concepts"])}

## Core Learnings
{bullet_list(book["core_learnings"])}

## Directions, Observations, And Implications
{bullet_list(book["decision_applications"])}

## How This Can Be Applied In Decision Making
{bullet_list(book["decision_applications"])}

## How This Strengthens A Bias-Aware RAG System
{bullet_list(book["ai_rag_relevance"])}
{chapter_section}

## Limitations And Cautions
{bullet_list(book["caution"])}

## Source Notes
{sources}

## RAG Ingestion Note
This document is intentionally written as a concise, retrieval-friendly synthesis rather than as a book review. It is designed to help the system retrieve:
- named biases and decision failures
- practical mitigation methods
- organizational and AI-governance implications
- scenario-to-concept mappings for user prompts
"""


def markdown_to_html(markdown_text: str, title: str) -> str:
    from markdown import markdown

    body = markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "nl2br"],
    )

    escaped_title = html.escape(title)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{escaped_title}</title>
  <style>
    body {{
      font-family: Georgia, 'Times New Roman', serif;
      margin: 48px 56px;
      color: #1f2933;
      line-height: 1.55;
      font-size: 12pt;
    }}
    h1 {{
      font-size: 24pt;
      margin-bottom: 0.35em;
      color: #0f172a;
    }}
    h2 {{
      font-size: 15pt;
      margin-top: 1.4em;
      color: #1d4ed8;
      border-bottom: 1px solid #dbe3ea;
      padding-bottom: 4px;
    }}
    p, li {{
      orphans: 3;
      widows: 3;
    }}
    ul {{
      margin-top: 0.3em;
    }}
    strong {{
      color: #111827;
    }}
    code {{
      font-family: Menlo, Monaco, monospace;
      font-size: 10pt;
      background: #f8fafc;
      padding: 1px 4px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def build_pdf(book: dict, pdf_path: Path) -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeader",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=colors.HexColor("#1D4ED8"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=book["title"],
        author=", ".join(book["authors"]),
    )

    story = [
        Paragraph(book["title"], styles["Title"]),
        Paragraph(
            f"<b>Authors:</b> {html.escape(', '.join(book['authors']))}<br/>"
            f"<b>Publication context:</b> {html.escape(book['year'])}<br/>"
            f"<b>Corpus priority:</b> {html.escape(book['priority_phase'])}<br/>"
            f"<b>Document type:</b> Synthesized RAG source dossier built from publicly available online material",
            styles["BodySmall"],
        ),
        Spacer(1, 6),
    ]

    sections = [
        ("Purpose In This RAG Corpus", book["why_important"]),
        ("Key Concepts And Retrieval Tags", book["key_concepts"]),
        ("Core Learnings", book["core_learnings"]),
        ("Directions, Observations, And Implications", book["decision_applications"]),
        ("How This Can Be Applied In Decision Making", book["decision_applications"]),
        ("How This Strengthens A Bias-Aware RAG System", book["ai_rag_relevance"]),
        ("Limitations And Cautions", book["caution"]),
    ]

    for section_title, items in sections:
        story.append(Paragraph(section_title, styles["SectionHeader"]))
        story.append(
            ListFlowable(
                [
                    ListItem(Paragraph(html.escape(item), styles["BodySmall"]))
                    for item in items
                ],
                bulletType="bullet",
                leftIndent=12,
            )
        )

    chapter_summaries = book.get("chapter_summaries", [])
    if chapter_summaries:
        story.append(Paragraph("Chapter-Wise Summary", styles["SectionHeader"]))
        story.append(
            ListFlowable(
                [
                    ListItem(
                        Paragraph(
                            f"<b>{html.escape(chapter['title'])}</b>: {html.escape(chapter['summary'])}",
                            styles["BodySmall"],
                        )
                    )
                    for chapter in chapter_summaries
                ],
                bulletType="bullet",
                leftIndent=12,
            )
        )

    story.append(Paragraph("Source Notes", styles["SectionHeader"]))
    story.append(
        ListFlowable(
            [
                ListItem(
                    Paragraph(
                        f"{html.escape(source['label'])} ({html.escape(source['type'])}): "
                        f"{html.escape(source['url'])}",
                        styles["BodySmall"],
                    )
                )
                for source in book["sources"]
            ],
            bulletType="bullet",
            leftIndent=12,
        )
    )

    story.append(Paragraph("RAG Ingestion Note", styles["SectionHeader"]))
    story.append(
        Paragraph(
            "This document is intentionally written as a concise, retrieval-friendly synthesis rather than as a book review. "
            "It is designed to help the system retrieve named biases and decision failures, practical mitigation methods, "
            "organizational and AI-governance implications, and scenario-to-concept mappings for user prompts.",
            styles["BodySmall"],
        )
    )

    doc.build(story)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(DEFAULT_BOOKS_FILE),
        help="Path to the JSON manifest describing book dossiers to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_MD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PDF_DIR.mkdir(parents=True, exist_ok=True)

    books_file = Path(args.input)
    books = json.loads(books_file.read_text(encoding="utf-8"))

    manifest = []
    for book in books:
      markdown_text = render_markdown(book)
      html_text = markdown_to_html(markdown_text, book["title"])

      md_path = OUTPUT_MD_DIR / f"{book['slug']}.md"
      html_path = OUTPUT_HTML_DIR / f"{book['slug']}.html"
      pdf_path = OUTPUT_PDF_DIR / f"{book['slug']}.pdf"

      md_path.write_text(markdown_text, encoding="utf-8")
      html_path.write_text(html_text, encoding="utf-8")
      build_pdf(book, pdf_path)

      manifest.append(
          {
              "title": book["title"],
              "slug": book["slug"],
              "markdown_path": str(md_path),
              "html_path": str(html_path),
              "pdf_path": str(pdf_path),
          }
      )

    (OUTPUT_HTML_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
