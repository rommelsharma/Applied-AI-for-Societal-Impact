"""
Generator for the Solution Design Document (.docx).

Produces ``docs/Solution Design Document.docx`` from this script. Re-run this
generator any time the canonical solution design changes; the .docx is the
output, this script is the source of truth.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "Solution Design Document.docx"


# -------------------------------------------------------------------------
# Color palette and typography helpers
# -------------------------------------------------------------------------

PRIMARY_BLUE = RGBColor(0x0F, 0x3D, 0x91)        # deep enterprise blue
ACCENT_BLUE = RGBColor(0x1D, 0x4E, 0xD8)         # mid blue for headings
DARK_GREY = RGBColor(0x1F, 0x29, 0x33)
MID_GREY = RGBColor(0x4B, 0x55, 0x63)
LIGHT_GREY_BG = "EEF2F7"                         # hex (no '#') for table shading
TABLE_HEADER_BG = "0F3D91"


def _set_cell_shading(cell, hex_fill: str) -> None:
    """Apply a fill colour to a table cell (python-docx has no first-class API)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def _set_cell_borders(cell, color="BFBFBF", size="6") -> None:
    """Apply consistent borders to all four sides of a cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), size)
        border.set(qn("w:color"), color)
        tc_borders.append(border)
    tc_pr.append(tc_borders)


def _add_horizontal_line(paragraph) -> None:
    """Insert a horizontal rule under a paragraph."""
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "BFBFBF")
    p_bdr.append(bottom)
    p_pr.append(p_bdr)


def _add_page_number_field(paragraph) -> None:
    """Inject a Word ``PAGE`` field so the footer auto-numbers."""
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr_text)
    run._r.append(fld_char2)


def configure_styles(doc: Document) -> None:
    """Set base typography for body, headings, and the title style."""
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.font.color.rgb = DARK_GREY

    for level, size, color in (
        (1, 18, PRIMARY_BLUE),
        (2, 14, ACCENT_BLUE),
        (3, 12, ACCENT_BLUE),
    ):
        style = doc.styles[f"Heading {level}"]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = color


# -------------------------------------------------------------------------
# Section primitives
# -------------------------------------------------------------------------

def add_heading(doc: Document, text: str, level: int = 1) -> None:
    """Add a heading and a subtle horizontal rule beneath level-1 headings."""
    paragraph = doc.add_heading(text, level=level)
    if level == 1:
        _add_horizontal_line(paragraph)


def add_paragraph(doc: Document, text: str, *, bold: bool = False, italic: bool = False) -> None:
    """Add a single body paragraph, optionally bold or italic."""
    paragraph = doc.add_paragraph()
    run = paragraph.add_run(text)
    run.bold = bold
    run.italic = italic


def add_bullets(doc: Document, items: list[str]) -> None:
    """Add a bullet list using Word's ``List Bullet`` style."""
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def add_numbered(doc: Document, items: list[str]) -> None:
    """Add a numbered list using Word's ``List Number`` style."""
    for item in items:
        doc.add_paragraph(item, style="List Number")


def add_definition_table(doc: Document, rows: list[tuple[str, str]]) -> None:
    """Add a two-column key/value table (term + definition / field + value)."""
    table = doc.add_table(rows=len(rows), cols=2)
    table.autofit = False
    table.columns[0].width = Cm(5.5)
    table.columns[1].width = Cm(11)

    for i, (key, value) in enumerate(rows):
        key_cell = table.cell(i, 0)
        val_cell = table.cell(i, 1)

        key_cell.text = ""
        val_cell.text = ""

        key_para = key_cell.paragraphs[0]
        key_run = key_para.add_run(key)
        key_run.bold = True
        key_run.font.color.rgb = PRIMARY_BLUE

        val_para = val_cell.paragraphs[0]
        val_para.add_run(value)

        for cell in (key_cell, val_cell):
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            _set_cell_borders(cell)
            _set_cell_shading(cell, LIGHT_GREY_BG if i % 2 == 0 else "FFFFFF")


def add_header_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    """Add a styled header-row + body-rows table."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.autofit = True

    header_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        header_cells[i].text = ""
        para = header_cells[i].paragraphs[0]
        run = para.add_run(header)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        _set_cell_shading(header_cells[i], TABLE_HEADER_BG)
        _set_cell_borders(header_cells[i], color="0F3D91")

    for r_idx, row in enumerate(rows, start=1):
        for c_idx, value in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = ""
            cell.paragraphs[0].add_run(value)
            _set_cell_borders(cell)
            _set_cell_shading(cell, LIGHT_GREY_BG if r_idx % 2 == 0 else "FFFFFF")


def add_diagram(doc: Document, lines: list[str]) -> None:
    """Add a monospaced ASCII diagram inside a single-cell shaded box."""
    table = doc.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)
    cell.text = ""
    para = cell.paragraphs[0]
    for i, line in enumerate(lines):
        run = para.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9)
        if i < len(lines) - 1:
            para.add_run().add_break()
    _set_cell_shading(cell, "F8FAFC")
    _set_cell_borders(cell, color="CBD5E1")


def add_page_break(doc: Document) -> None:
    """Insert a hard page break."""
    doc.add_page_break()


# -------------------------------------------------------------------------
# Document sections
# -------------------------------------------------------------------------

def build_cover_page(doc: Document) -> None:
    """Construct the cover page: title, subtitle, classification, date."""
    section = doc.sections[0]
    section.top_margin = Cm(2.4)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)

    doc.add_paragraph()
    doc.add_paragraph()

    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run("Solution Design Document")
    title_run.bold = True
    title_run.font.size = Pt(32)
    title_run.font.color.rgb = PRIMARY_BLUE

    subtitle_para = doc.add_paragraph()
    subtitle_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_run = subtitle_para.add_run("Bias-Aware Decision Making RAG Copilot")
    subtitle_run.font.size = Pt(18)
    subtitle_run.font.color.rgb = ACCENT_BLUE

    domain_para = doc.add_paragraph()
    domain_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    domain_run = domain_para.add_run("Decision Intelligence | Applied AI for Societal Impact")
    domain_run.italic = True
    domain_run.font.size = Pt(12)
    domain_run.font.color.rgb = MID_GREY

    for _ in range(8):
        doc.add_paragraph()

    add_definition_table(
        doc,
        [
            ("Document Title", "Solution Design Document - Bias-Aware Decision Making RAG Copilot"),
            ("Project", "Applied-AI-for-Societal-Impact / decision_intelligence"),
            ("Author", "Rommel Sharma"),
            ("Role", "Enterprise Technology Leader | Applied AI Practitioner"),
            ("Document Status", "Draft for Review"),
            ("Version", "1.0"),
            ("Issue Date", date.today().strftime("%d %B %Y")),
            ("Classification", "Portfolio / Public"),
        ],
    )

    add_page_break(doc)


def build_document_control(doc: Document) -> None:
    """Add document-control and revision-history tables."""
    add_heading(doc, "Document Control", level=1)

    add_heading(doc, "Revision History", level=2)
    add_header_table(
        doc,
        ["Version", "Date", "Author", "Summary of Changes"],
        [
            ["0.1", "Initial", "Rommel Sharma", "Document scaffolding aligned with repo state."],
            ["1.0", date.today().strftime("%d %b %Y"), "Rommel Sharma", "Baseline solution design document for portfolio review."],
        ],
    )

    add_heading(doc, "Reviewers and Approvers", level=2)
    add_header_table(
        doc,
        ["Name", "Role", "Responsibility"],
        [
            ["Rommel Sharma", "Author / Architect", "End-to-end design, implementation, and validation"],
            ["TBD", "Technical Reviewer", "Architecture and security review"],
            ["TBD", "Domain Reviewer", "Bias-taxonomy completeness and decision-science accuracy"],
        ],
    )

    add_heading(doc, "Related Documents", level=2)
    add_header_table(
        doc,
        ["Document", "Location", "Purpose"],
        [
            ["README.md", "<root>/README.md", "Project overview and current status"],
            ["docs/code-flow.md", "<root>/docs/code-flow.md", "Directory structure and code flow"],
            ["docs/fundamental-concepts.md", "<root>/docs/fundamental-concepts.md", "Technology choices and rationale"],
            ["docs/ARCHITECTURE.md", "<root>/docs/ARCHITECTURE.md", "Architecture summary"],
            ["docs/DATA_MODEL.md", "<root>/docs/DATA_MODEL.md", "Data model contracts"],
            ["docs/NEXT_STEPS.md", "<root>/docs/NEXT_STEPS.md", "Forward roadmap"],
        ],
    )

    add_page_break(doc)


def build_executive_summary(doc: Document) -> None:
    """Top-level summary of problem, approach, and value proposition."""
    add_heading(doc, "1. Executive Summary", level=1)

    add_paragraph(
        doc,
        "The Bias-Aware Decision Making RAG Copilot is a portfolio-grade applied-AI "
        "system that helps decision makers reason more carefully about complex, "
        "human-centred situations. It identifies likely cognitive and systemic biases, "
        "surfaces noise (inconsistency in judgment), and grounds its analysis in a "
        "curated knowledge base of seminal decision-science literature.",
    )

    add_paragraph(
        doc,
        "The solution combines three layers: (1) a deliberately curated and structured "
        "knowledge corpus, (2) a Retrieval-Augmented Generation (RAG) pipeline backed by "
        "Amazon Bedrock embeddings and FAISS vector search, and (3) a strict-schema "
        "reasoning layer powered by Claude Sonnet 4.5. Every scenario is processed "
        "twice - once without retrieval (baseline) and once with retrieval (RAG-enhanced) - "
        "so the lift produced by retrieval can be measured rather than assumed.",
    )

    add_paragraph(
        doc,
        "The system is positioned explicitly as decision support, not legal, medical, "
        "or HR advice. It is engineered for explainability, source traceability, and "
        "auditability from raw PDF through to final response payload.",
    )

    add_heading(doc, "Key Outcomes", level=2)
    add_bullets(
        doc,
        [
            "End-to-end ingestion pipeline covering parsing, chunking, taxonomy-driven concept extraction, enrichment, embedding, and indexing.",
            "Production-style provider abstraction over Amazon Bedrock (Claude Sonnet 4.5 for chat, Titan v2 for embeddings).",
            "FAISS-backed retrieval layer with concept and decision-domain filters and an unfiltered fallback path.",
            "Comparison-mode bias detection that returns baseline and RAG-enhanced answers in one call, with full retrieval traceability.",
            "Evaluation harness over ten realistic decision-making scenarios (hiring, performance review, automation bias, groupthink, base-rate neglect, etc.).",
            "Documentation set engineered for reviewer scrutiny: project context, architecture, data model, code flow, fundamental concepts, and this design document.",
        ],
    )

    add_page_break(doc)


def build_background(doc: Document) -> None:
    """Background, problem statement, and target users."""
    add_heading(doc, "2. Background and Context", level=1)

    add_heading(doc, "2.1 Problem Statement", level=2)
    add_paragraph(
        doc,
        "Human judgment in high-stakes situations - hiring, performance review, "
        "compliance investigation, leadership decision-making - is systematically "
        "distorted by cognitive biases (anchoring, confirmation, similarity, halo) "
        "and noise (inconsistency between reviewers and across occasions). The "
        "behavioural-science literature has documented these failure modes for "
        "decades, but the knowledge is fragmented across many books and rarely "
        "available at the moment of decision.",
    )

    add_heading(doc, "2.2 Opportunity", level=2)
    add_paragraph(
        doc,
        "A retrieval-augmented AI assistant can act as a structured second opinion: "
        "scan the situation, surface the biases that are most likely at play, ground "
        "the analysis in published research, and recommend mitigations. The role is "
        "support, not authority - the goal is better human judgment, not delegated "
        "decisions.",
    )

    add_heading(doc, "2.3 Target Users", level=2)
    add_bullets(
        doc,
        [
            "Managers and team leaders evaluating people, projects, and trade-offs.",
            "HR and recruiting professionals running structured interviews and performance reviews.",
            "Legal, compliance, and policy reviewers assessing case patterns.",
            "Individual contributors making difficult judgment calls.",
            "Portfolio reviewers and recruiters evaluating applied-AI capability.",
        ],
    )

    add_heading(doc, "2.4 Disclaimer and Boundaries", level=2)
    add_paragraph(
        doc,
        "The system is not legal, medical, financial, or HR-policy advice. It "
        "provides research-informed perspectives and structured thinking guidance "
        "and does not replace professional judgment or organisational policy.",
        italic=True,
    )

    add_page_break(doc)


def build_objectives(doc: Document) -> None:
    """Objectives, scope, and non-goals."""
    add_heading(doc, "3. Objectives and Scope", level=1)

    add_heading(doc, "3.1 Objectives", level=2)
    add_numbered(
        doc,
        [
            "Build a credible, traceable RAG system grounded in curated decision-science literature.",
            "Demonstrate measurable lift from retrieval over a baseline LLM call.",
            "Provide structured, auditable JSON outputs that downstream UIs and evaluators can render deterministically.",
            "Use enterprise-grade patterns: centralised configuration, central path resolution, source attribution, observability hooks.",
            "Keep the corpus, taxonomy, and prompt out of code so iteration does not require redeployment.",
        ],
    )

    add_heading(doc, "3.2 In Scope", level=2)
    add_bullets(
        doc,
        [
            "PDF ingestion, parsing, and metadata extraction.",
            "Chunking with front- and back-matter cleanup.",
            "Taxonomy-driven concept extraction and enrichment.",
            "Embedding via Amazon Titan v2 and indexing via FAISS.",
            "Concept- and domain-filtered retrieval.",
            "Baseline-vs-RAG bias detection through Claude Sonnet 4.5.",
            "Evaluation harness with persisted comparison results.",
            "Auxiliary tooling for generating retrieval-friendly book dossiers.",
        ],
    )

    add_heading(doc, "3.3 Out of Scope (Current Phase)", level=2)
    add_bullets(
        doc,
        [
            "Authenticated multi-user UI with persistence.",
            "Production-grade fine-tuning of foundation models.",
            "Real-time conversational memory across sessions.",
            "Direct integration with HRIS, ATS, or case-management systems.",
            "Automated decision-making (the system is decision support only).",
        ],
    )

    add_page_break(doc)


def build_solution_overview(doc: Document) -> None:
    """High-level solution overview with the principal flow diagram."""
    add_heading(doc, "4. Solution Overview", level=1)

    add_paragraph(
        doc,
        "At runtime a user submits a free-text scenario. The system extracts taxonomy "
        "concepts from that scenario, retrieves the most relevant chunks from the "
        "curated knowledge base, and runs the chat model twice with the same locked "
        "JSON-schema prompt - once without retrieved context (baseline) and once "
        "with retrieved context (RAG-enhanced). Both answers plus a traceability "
        "record of which chunks were retrieved are returned in a single payload.",
    )

    add_heading(doc, "4.1 Principal Logical Flow", level=2)
    add_diagram(
        doc,
        [
            "                  +----------------------------+",
            "                  |    User scenario (text)    |",
            "                  +-------------+--------------+",
            "                                |",
            "                +---------------+----------------+",
            "                |                                |",
            "                v                                v",
            "      [BASELINE PATH]                    [RAG PATH]",
            "      taxonomy + LLM                taxonomy + retrieval + LLM",
            "                |                                |",
            "                |                +---------------+--------------+",
            "                |                | concept extraction on text   |",
            "                |                | embed scenario (Titan v2)    |",
            "                |                | FAISS top-k similarity       |",
            "                |                | concept / domain filters     |",
            "                |                | format context blocks        |",
            "                |                +---------------+--------------+",
            "                |                                |",
            "                v                                v",
            "        Claude Sonnet 4.5                Claude Sonnet 4.5",
            "      (no retrieved context)            (with retrieved context)",
            "                |                                |",
            "                v                                v",
            "         Strict JSON answer              Strict JSON answer",
            "                |                                |",
            "                +-------------+------------------+",
            "                              v",
            "  Comparison payload {without_rag, with_rag, retrieval}",
        ],
    )

    add_heading(doc, "4.2 Principal Components", level=2)
    add_header_table(
        doc,
        ["Layer", "Module", "Responsibility"],
        [
            ["Configuration", "shared_components/settings.py", "Single source of truth for environment-driven configuration"],
            ["Paths", "shared_components/utilities/path_utils.py", "Centralised resolution of every project directory"],
            ["Taxonomy", "shared_components/utilities/taxonomy_utils.py", "Cached loaders for bias taxonomy and support concepts"],
            ["Ingestion", "data_pipeline/*.py", "Parse, chunk, tag, enrich, and embed the corpus"],
            ["Retrieval", "rag/retriever.py", "FAISS-backed vector search with optional filters"],
            ["Inference", "app/services/bedrock_provider.py", "Bedrock client for chat completion and embeddings"],
            ["Orchestration", "app/services/bias_detector.py", "Baseline-vs-RAG comparison with strict JSON output"],
            ["Evaluation", "evaluation/*", "Test scenarios and evaluation runner"],
        ],
    )

    add_page_break(doc)


def build_detailed_design(doc: Document) -> None:
    """Detailed design covering ingestion pipeline, runtime path, data model."""
    add_heading(doc, "5. Detailed Design", level=1)

    add_heading(doc, "5.1 Offline Ingestion Pipeline", level=2)
    add_paragraph(
        doc,
        "The offline pipeline transforms raw PDFs into a vector-searchable, "
        "traceable knowledge base in five sequential stages.",
    )

    add_diagram(
        doc,
        [
            "PDFs (data/raw/)",
            "   |",
            "   v",
            "[1] pdf_parser.py        ->  parsed_text/{slug}.txt + parsed_text/{slug}.meta.json",
            "   |",
            "   v",
            "[2] chunker.py           ->  chunks/chunks.json",
            "   |",
            "   v",
            "[3] concept_extractor.py ->  chunks/chunks_with_concepts.json",
            "   |",
            "   v",
            "[4] enrich_chunks.py     ->  knowledge/knowledge_base.json",
            "   |",
            "   v",
            "[5] build_vector_index.py -> vector_store/{embeddings.npy, knowledge.index,",
            "                             index_metadata.json, manifest.json}",
        ],
    )

    add_paragraph(doc, "Stage responsibilities:", bold=True)
    add_bullets(
        doc,
        [
            "Stage 1 (PDF parsing): PyMuPDF text extraction plus regex-based metadata heuristics for author, title, and publication year. Records ``parser_notes`` whenever a field cannot be inferred.",
            "Stage 2 (Chunking): trims known front-matter and back-matter markers past appropriate floors, normalises whitespace, and splits the body into ~350-word chunks with 60-word overlap. Each chunk inherits source metadata and gets a stable id.",
            "Stage 3 (Concept extraction): builds a unified concept catalog from ``bias-taxonomy.json`` and ``retrieval-concepts.json`` and tags each chunk with concepts whose canonical name or aliases appear in the text.",
            "Stage 4 (Enrichment): generates summaries, derives importance and decision_domains from concept metadata, and extracts top keywords. Output is the retrieval-ready ``knowledge_base.json``.",
            "Stage 5 (Vector index build): embeds each chunk via Amazon Titan v2 (1024 dims, normalised), persists ``embeddings.npy``, builds a FAISS ``IndexFlatIP``, and writes aligned ``index_metadata.json`` plus a ``manifest.json``.",
        ],
    )

    add_heading(doc, "5.2 Runtime Path", level=2)
    add_paragraph(
        doc,
        "At runtime, ``app/services/bias_detector.detect_bias_comparison`` orchestrates "
        "the entire flow.",
    )

    add_diagram(
        doc,
        [
            "User scenario",
            "   |",
            "   v",
            "[A] extract_concepts(scenario)            <- taxonomy-driven",
            "   |",
            "   v",
            "[B] KnowledgeRetriever.search(scenario, preferred_concepts=...)",
            "        - embed via Titan v2",
            "        - FAISS top_k * 3 candidates",
            "        - concept / domain filters",
            "        - unfiltered fallback if filters strip everything",
            "   |",
            "   v",
            "[C] format_retrieved_context(results)     <- numbered, traceable blocks",
            "   |",
            "   v",
            "[D] build_system_prompt(...)              <- baseline and RAG variants",
            "   |",
            "   v",
            "[E] BedrockProvider.converse() x 2        <- Claude Sonnet 4.5",
            "   |",
            "   v",
            "[F] {without_rag, with_rag, retrieval}",
        ],
    )

    add_heading(doc, "5.3 Data Model and Traceability Contract", level=2)
    add_header_table(
        doc,
        ["Field", "Origin", "Purpose"],
        [
            ["id", "chunker", "Stable identifier ``{slug}_chunk_{idx}``"],
            ["source / file_name / author / title / publication_year", "pdf_parser", "Source attribution carried through every step"],
            ["parser_notes", "pdf_parser", "Records fields that could not be inferred"],
            ["cleaning_notes", "chunker", "Records front/back-matter trimming events"],
            ["concepts / concept_confidence", "concept_extractor", "Symbolic taxonomy tags"],
            ["importance / decision_domains / keywords", "enrich_chunks", "Retrieval-time filters and signals"],
            ["text / summary", "chunker / enrich_chunks", "What the LLM ultimately sees in the system prompt"],
            ["score", "retriever", "Similarity to the query (cosine via inner product on unit vectors)"],
        ],
    )

    add_heading(doc, "5.4 Prompt Design", level=2)
    add_paragraph(
        doc,
        "The system prompt (``prompts/bias_detection_system_prompt.txt``) locks the "
        "model into a strict JSON contract: situation_summary, biases_identified "
        "(each with bias_name, bias_layer, confidence, explanation, evidence, risk), "
        "noise_considerations, overall_risk_assessment, recommended_actions, and "
        "alternative_perspectives. The full bias taxonomy is appended to the prompt "
        "so the model is restricted to a closed vocabulary and cannot invent biases.",
    )

    add_page_break(doc)


def build_key_concepts(doc: Document) -> None:
    """Technology choices and rationale."""
    add_heading(doc, "6. Key Concepts and Technology Choices", level=1)

    add_heading(doc, "6.1 Retrieval-Augmented Generation (RAG)", level=2)
    add_paragraph(
        doc,
        "RAG enriches an LLM's prompt at runtime with passages retrieved from an "
        "operator-controlled knowledge base. This project uses RAG to deliver three "
        "properties a vanilla LLM cannot provide: source attribution (every claim "
        "traces to a named book/author/chunk), curatorial control (the corpus is "
        "deliberately selected behavioural-science literature), and update without "
        "retraining (adding a new book is a corpus operation, not a model operation).",
    )

    add_heading(doc, "6.2 Embeddings", level=2)
    add_paragraph(
        doc,
        "An embedding is a fixed-length numeric vector that represents the meaning "
        "of a text snippet. Two snippets with similar meaning produce vectors that "
        "point in similar directions in high-dimensional space, even when they share "
        "no exact words. User scenarios are paraphrases - they never quote the books - "
        "so semantic search via embeddings is the only retrieval strategy that "
        "survives paraphrase.",
    )

    add_heading(doc, "6.3 Why Amazon Titan Text Embeddings v2", level=2)
    add_bullets(
        doc,
        [
            "Hosted, managed, consistent with the inference plane (Bedrock) - keeps the operational footprint small.",
            "1024-dimension vectors with ``normalize=true`` so inner-product similarity equals cosine similarity (matches FAISS IndexFlatIP semantics).",
            "Removes the need to host or fine-tune a separate embedding model for a portfolio system.",
            "Provider swap is a one-line change because all embedding access flows through ``BedrockProvider.embed_text()``.",
        ],
    )

    add_heading(doc, "6.4 Why FAISS (IndexFlatIP)", level=2)
    add_paragraph(
        doc,
        "FAISS (Facebook AI Similarity Search) is the de-facto standard library for "
        "similarity search over dense vectors. ``IndexFlatIP`` is exhaustive, exact "
        "(zero approximation error), and microsecond-fast at the project's current "
        "corpus size. Combined with normalised Titan embeddings, inner-product "
        "search is mathematically equivalent to cosine similarity. The retriever "
        "depends only on the ``search(query_vector, k)`` interface, so the index "
        "can later be swapped for ``IndexIVFFlat`` or HNSW without touching any "
        "other code. A pure-numpy fallback path is preserved for environments "
        "without FAISS installed.",
    )

    add_heading(doc, "6.5 Why Amazon Bedrock", level=2)
    add_bullets(
        doc,
        [
            "One control plane for chat (Claude) and embeddings (Titan) - same client, same IAM, same VPC patterns.",
            "Enterprise posture: VPC endpoints, KMS, CloudTrail, no model-vendor sprawl, no separate billing.",
            "Model swap without code changes: only the model id in ``.env`` changes.",
            "Bedrock does not use customer prompts to train foundation models - important for sensitive scenarios.",
            "Aligned with the AWS-native stack assumed for hosting (S3 for vector store, CloudWatch for observability, IAM for auth).",
        ],
    )

    add_heading(doc, "6.6 Why Claude Sonnet 4.5", level=2)
    add_bullets(
        doc,
        [
            "Faithful adherence to a strict JSON schema and a closed taxonomy.",
            "Long, structured output without truncation.",
            "Nuanced reasoning across human and systemic factors in a scenario.",
            "Consistency at temperature ``0.1`` so baseline-vs-RAG comparisons are reproducible.",
        ],
    )

    add_heading(doc, "6.7 Why a Taxonomy-Driven Concept Layer", level=2)
    add_paragraph(
        doc,
        "Pure-vector RAG is fragile: unrelated chunks can embed close together and "
        "the LLM has no structured handle on what kind of bias a chunk is about. "
        "The taxonomy and support-concept catalog are JSON files used at indexing "
        "time (chunk tagging), at retrieval time (filters), and at generation time "
        "(closed vocabulary in the system prompt). This hybrid (semantic + symbolic) "
        "approach is central to the explainability story.",
    )

    add_page_break(doc)


def build_validation_qa(doc: Document) -> None:
    """Validation, evaluation, and QA strategy."""
    add_heading(doc, "7. Validation and QA Strategy", level=1)

    add_heading(doc, "7.1 Evaluation Architecture", level=2)
    add_paragraph(
        doc,
        "Every scenario is evaluated through the same comparison harness. The "
        "harness runs ``detect_bias_comparison``, persists the full payload "
        "(without_rag, with_rag, retrieval) to ``evaluation/latest_results.json``, "
        "and prints lightweight diff statistics. This is the seed for richer "
        "scorecards.",
    )

    add_heading(doc, "7.2 Evaluation Dimensions (Roadmap)", level=2)
    add_header_table(
        doc,
        ["Dimension", "What it Measures", "How"],
        [
            ["Schema Fidelity", "Whether the model produced valid JSON matching the contract", "Strict parse via ``extract_json_payload`` plus schema validator"],
            ["Taxonomy Adherence", "Whether bias_name values come from the closed taxonomy", "Set membership check against the taxonomy keys"],
            ["Grounding", "Whether the RAG path's claims are supported by retrieved chunks", "LLM-as-judge or n-gram overlap against retrieved excerpts"],
            ["Retrieval Relevance", "Whether retrieved chunks address the scenario's biases", "Concept overlap and human spot-checks on top-k results"],
            ["Lift", "Quality difference between baseline and RAG outputs", "Side-by-side bias counts, evidence specificity, and reviewer rubric"],
            ["Determinism", "Consistency of output across runs at temperature 0.1", "Repeated calls and diff comparison"],
        ],
    )

    add_heading(doc, "7.3 Test Scenarios", level=2)
    add_paragraph(
        doc,
        "``evaluation/test_scenarios.json`` covers ten realistic situations across "
        "hiring (similarity bias, halo, unstructured culture-fit), performance "
        "review (level noise across reviewers), strategy (sunk-cost, availability "
        "via competitor failure), AI/ML (historical-data bias, automation bias), "
        "compliance (representativeness vs base rates), and team dynamics "
        "(groupthink). The set is intentionally small but high-signal; expansion "
        "is part of the roadmap.",
    )

    add_heading(doc, "7.4 Quality Assurance Practices", level=2)
    add_bullets(
        doc,
        [
            "Static configuration validation: ``BedrockProvider`` fails fast at construction if credentials are missing.",
            "JSON-malformed retry: a tighter retry path catches near-miss outputs at the token boundary.",
            "Filter fallback: retrieval never returns empty - if filters strip everything, an unfiltered semantic search is run.",
            "Traceability checks: every retrieval result carries id, source, author, score, concepts, and decision_domains so any answer can be audited.",
            "Reproducibility: ``temperature=0.1`` and a frozen taxonomy reduce run-to-run variability.",
        ],
    )

    add_page_break(doc)


def build_infrastructure(doc: Document) -> None:
    """Infrastructure, hosting, and security."""
    add_heading(doc, "8. Infrastructure and Hosting", level=1)

    add_heading(doc, "8.1 Reference AWS Architecture", level=2)
    add_diagram(
        doc,
        [
            "                                    +-------------------------+",
            "                                    |   End User / Recruiter  |",
            "                                    +-----------+-------------+",
            "                                                |",
            "                                                v",
            "                                    +-------------------------+",
            "                                    |     CloudFront (CDN)    |",
            "                                    +-----------+-------------+",
            "                                                |",
            "                                                v",
            "                                    +-------------------------+",
            "                                    |  API Gateway / ALB      |",
            "                                    +-----------+-------------+",
            "                                                |",
            "                                                v",
            "                +---------------+   +-------------------------+   +---------------+",
            "                | AWS Secrets   |-->|  Service (ECS Fargate / |<--|  Amazon S3    |",
            "                |   Manager     |   |   Lambda) - Copilot API |   |  vector store |",
            "                +---------------+   +-----------+-------------+   +---------------+",
            "                                                |",
            "                                                v",
            "                                    +-------------------------+",
            "                                    |      Amazon Bedrock     |",
            "                                    |  Claude 4.5  +  Titan v2|",
            "                                    +-----------+-------------+",
            "                                                |",
            "                                                v",
            "                                    +-------------------------+",
            "                                    |  CloudWatch Logs/Metrics|",
            "                                    |  X-Ray  +  CloudTrail   |",
            "                                    +-------------------------+",
        ],
    )

    add_heading(doc, "8.2 Component Choices", level=2)
    add_header_table(
        doc,
        ["Concern", "Service", "Why"],
        [
            ["Foundation models", "Amazon Bedrock", "One control plane, no data leakage, IAM-aligned"],
            ["Vector storage", "Amazon S3 + FAISS artefacts", "Cheap, durable, region-local; FAISS loaded into compute memory"],
            ["Compute", "AWS Fargate (ECS) or AWS Lambda (container)", "No server management; right-sized for low-QPS demo workloads"],
            ["API edge", "API Gateway (HTTP API) or ALB", "Auth, throttling, observability"],
            ["CDN / static UI", "Amazon CloudFront + S3", "Static React/Streamlit-export front end"],
            ["Secrets", "AWS Secrets Manager", "Rotates Bedrock bearer tokens and DB creds"],
            ["Networking", "VPC + Bedrock VPC endpoint", "No public internet on the inference path"],
            ["Logs and metrics", "Amazon CloudWatch + X-Ray", "Request tracing across retrieval and inference"],
            ["Audit", "AWS CloudTrail", "Immutable record of every Bedrock call"],
            ["IAM", "Least-privilege roles per workload", "Separate roles for ingestion, inference, and admin"],
        ],
    )

    add_heading(doc, "8.3 Deployment Topology", level=2)
    add_bullets(
        doc,
        [
            "Single-region deployment in ``us-west-2`` (Bedrock model availability and latency).",
            "Two environments: ``dev`` (single AZ, smaller Fargate task) and ``prod`` (multi-AZ, ALB with health checks).",
            "Vector store artefacts live in S3 under ``s3://<bucket>/vector_store/`` with versioning enabled.",
            "Container images stored in Amazon ECR with image-scanning enabled.",
            "Model invocation goes through a Bedrock VPC endpoint - no public egress required.",
        ],
    )

    add_heading(doc, "8.4 Security and Compliance Considerations", level=2)
    add_bullets(
        doc,
        [
            "All inbound traffic is HTTPS-only; ALB / API Gateway terminates TLS with ACM certificates.",
            "Bedrock bearer token stored in Secrets Manager and retrieved at container start.",
            "All data at rest encrypted with AWS KMS (S3 SSE-KMS, EBS, Secrets Manager).",
            "Inputs are not logged in clear text; CloudWatch logs redact scenario text by default.",
            "Bedrock does not use prompts for foundation-model training, so user content remains private.",
            "IAM roles follow least privilege: ingestion role can write to S3 vector store but not invoke Bedrock chat; inference role can invoke Bedrock but not write S3.",
        ],
    )

    add_page_break(doc)


def build_mlops(doc: Document) -> None:
    """ML-Ops best practices, operations, and governance.

    The full plan lives in ``docs/ml-ops.plan.md``; this section is the
    design-document-grade summary.
    """
    add_heading(doc, "9. ML-Ops Best Practices", level=1)

    add_paragraph(
        doc,
        "A RAG system has three artefacts that move on independent clocks: "
        "code, corpus, and foundation models. Conventional MLOps assumes a "
        "trained model; here the LLMs are pre-trained and the operational "
        "lever is the corpus and the prompt. The practices below treat the "
        "corpus as a first-class artefact with the same versioning, "
        "evaluation, and deployment discipline normally applied to model "
        "weights. The complete plan is maintained in docs/ml-ops.plan.md.",
    )

    add_heading(doc, "9.1 Reproducibility", level=2)
    add_bullets(
        doc,
        [
            "Pinned dependency floors in requirements.txt (boto3, faiss-cpu, numpy, PyMuPDF, python-dotenv, python-docx).",
            "shared_components/settings.py is the single module that reads environment variables; every other file imports BEDROCK_SETTINGS and RAG_SETTINGS.",
            "shared_components/utilities/path_utils.py is the single module that resolves filesystem paths; corpus selection is controlled by CORPUS_NAME.",
            "Chunking parameters (PROFILES, _FRONT_HEADING, _BACK_HEADING, _TRIM_SAFETY_THRESHOLD) are module-level constants.",
            "Taxonomy and support-concepts JSON files are committed under data/metadata/ so the symbolic layer is fully auditable.",
            "Bedrock chat is invoked at temperature 0.1; Bedrock Titan v2 embeddings are deterministic; FAISS IndexFlatIP returns ranked candidates in stable order.",
            "Each vector index writes manifest.json (corpus name, count, dimension, index type, chunking-profile distribution); the runtime can refuse to start on a mismatch.",
        ],
    )

    add_heading(doc, "9.2 Versioning of Independent Lifecycles", level=2)
    add_header_table(
        doc,
        ["Versioned artefact", "Where it lives", "How it advances"],
        [
            ["Code", "git commit hash", "Per pull request"],
            ["Bedrock chat model", "BEDROCK_MODEL_ID in .env", "Manual env change + redeploy"],
            ["Bedrock embedding model", "BEDROCK_EMBEDDING_MODEL_ID in .env", "Forces a full re-embed of the corpus"],
            ["Bias taxonomy", "data/metadata/bias-taxonomy.json (committed)", "Pull request with reviewer sign-off"],
            ["Corpus snapshot", "vector_store/manifest.json + git tag of build commit", "Per ingestion run"],
        ],
    )

    add_heading(doc, "9.3 Data and Model Lineage", level=2)
    add_paragraph(
        doc,
        "Every chunk in the knowledge base carries a complete chain of custody, "
        "from raw PDF through final response payload:",
    )
    add_diagram(
        doc,
        [
            "  raw PDF",
            "    +- source_filename, sha256 (planned)",
            "    +- parsed_text + .meta.json (parser_notes for any field that could not be inferred)",
            "        +- chunks.json (cleaning_notes, chunking_profile, chapter_title)",
            "            +- chunks_with_concepts.json (concept tags, concept_confidence)",
            "                +- knowledge_base.json (passage_type, decision_phase, importance,",
            "                                       decision_domains, summary, keywords, classifier)",
            "                    +- embeddings.npy + index_metadata.json (1:1 row alignment)",
            "                        +- retrieval result (similarity score, chunk id, source,",
            "                                           chapter, concepts, passage_type)",
            "                            +- bias_detector response payload",
            "                                (retrieval array preserved alongside with_rag JSON)",
        ],
    )

    add_heading(doc, "9.4 CI/CD Strategy", level=2)
    add_paragraph(doc, "Code pipeline (continuous integration):")
    add_bullets(
        doc,
        [
            "lint (ruff/black) - unit tests - schema tests on taxonomy / concepts JSON - smoke ingest on a synthetic PDF - container build with Trivy scan - push to ECR on merge.",
        ],
    )
    add_paragraph(doc, "Corpus pipeline (continuous corpus delivery):")
    add_bullets(
        doc,
        [
            "JSON schema validate - regenerate dossier PDFs - python data_pipeline/build_knowledge_base.py --corpus public - python data_pipeline/build_vector_index.py --corpus public.",
            "Run python evaluation/run_evaluation.py to produce a comparison report against the previous baseline; regressions block merge.",
            "Upload the new vector store to S3 with a versioned object key.",
        ],
    )
    add_paragraph(doc, "Deployment:")
    add_bullets(
        doc,
        [
            "Blue/green deployment via ECS service-update or Lambda alias-shift on traffic.",
            "Manual approval gate for production until automated evaluation gating reaches the agreed quality bar.",
            "Vector-store hot reload (planned): runtime watches a versioned S3 prefix and atomically swaps to the new manifest, no redeploy needed for corpus refresh.",
        ],
    )

    add_heading(doc, "9.5 Evaluation Gating", level=2)
    add_paragraph(
        doc,
        "The deployment pipeline runs the evaluation harness against the "
        "canonical 30-plus scenarios on every release candidate. Releases are "
        "blocked when:",
    )
    add_bullets(
        doc,
        [
            "Schema fidelity drops below 100 percent on the canonical scenarios.",
            "Taxonomy adherence drops below 100 percent (any invented bias name is a hard fail).",
            "Bias-count delta between baseline and RAG indicates a regression.",
            "P95 end-to-end latency exceeds the agreed SLO (currently 8 seconds).",
            "Empty-retrieval fraction exceeds 1 percent (early signal of corpus or query-encoder drift).",
        ],
    )
    add_paragraph(
        doc,
        "A nightly job runs scripts/compare_versions.py --versions public,private "
        "so the lift between corpora is monitored as a moving baseline rather "
        "than a one-off measurement.",
    )

    add_heading(doc, "9.6 Observability", level=2)
    add_bullets(
        doc,
        [
            "Structured JSON logs for every retrieval and inference call: trace_id, scenario_id, corpus, top_k, mmr_lambda, retrieved_chunk_ids, similarity_scores, baseline_call_ms, rag_call_ms, json_parse_retries, input_tokens, output_tokens, model_id.",
            "CloudWatch metrics: P50/P95 retrieval latency, P50/P95 inference latency for each path, JSON-parse retry rate, empty-result rate, Bedrock throttling rate, estimated cost per scenario.",
            "Distributed tracing (AWS X-Ray) wraps embed_query, faiss_search, mmr, rerank, format_context, baseline_call, rag_call.",
            "Alarms: Bedrock throttling > 1 percent over 5 minutes, JSON retry > 5 percent, empty-result > 5 percent, P95 latency > 12 seconds, > 30 percent change in average bias count vs trailing 7-day baseline.",
        ],
    )

    add_heading(doc, "9.7 Cost Management", level=2)
    add_bullets(
        doc,
        [
            "Embeddings cached on disk (embeddings.npy) and reused across queries; only the query is embedded at runtime.",
            "Two-layer retrieval - FAISS pulls a wide pool, the optional reranker scores at most a fixed top-N (default 24).",
            "MMR runs before the reranker so the LLM judge sees a constant number of candidates regardless of corpus size.",
            "LLM-based passage classification (ENRICHMENT_USE_LLM) is off by default; rule-based classifier is free.",
            "Claude-Haiku reranker (RAG_RERANKER) is off by default; documented but not enabled until corpus exceeds ~3,000 chunks.",
            "Monthly cost dashboard splits spend by chat, embedding-runtime, embedding-corpus-build, and rerank.",
        ],
    )

    add_heading(doc, "9.8 Drift Monitoring", level=2)
    add_bullets(
        doc,
        [
            "Corpus drift: chunk counts and concept distributions tracked between builds; > 10 percent shift opens a review issue.",
            "Retrieval-quality drift: same canonical query suite runs nightly; significant changes in average similarity score or top-k chunk identity flag a regression.",
            "Output drift: bias-count and recommended-action distributions per domain are tracked; abrupt changes are escalated (a vendor model upgrade often presents this way).",
        ],
    )

    add_heading(doc, "9.9 Disaster Recovery", level=2)
    add_header_table(
        doc,
        ["Artefact", "Backup", "RPO / RTO target"],
        [
            ["Vector store (public)", "Versioned S3 bucket with object-lock", "RPO 24h / RTO 1h"],
            ["Taxonomies and prompts", "git history", "RPO 0 / RTO < 15 min"],
            ["Container images", "ECR with lifecycle policy retaining last 30 images", "RPO 0 / RTO < 30 min"],
            ["Secrets", "AWS Secrets Manager rotation history", "RPO 0 / RTO < 30 min"],
            ["Private corpus", "Local-laptop backup (Time Machine or equivalent)", "Out of production DR scope"],
        ],
    )
    add_paragraph(
        doc,
        "A failed Bedrock region is mitigated by Bedrock cross-region failover "
        "(AWS-managed) and, for a future hardening, a secondary Bedrock region "
        "in us-east-1 configured as warm standby.",
    )

    add_heading(doc, "9.10 Responsible AI and Governance", level=2)
    add_bullets(
        doc,
        [
            "Disclaimer surfaced in every UI and in the response payload: not legal, medical, HR, or financial advice.",
            "Closed-vocabulary output - the taxonomy is injected into the system prompt and the LLM cannot invent bias names.",
            "Source attribution - every RAG response carries a retrieval array with chunk ids, sources, similarity scores, and concepts.",
            "Human-in-the-loop is the explicit operating model; the system never automates a decision.",
            "Coverage and limitations documented in the README so consumers understand the cultural and linguistic lens.",
            "Audit-ready history retained under data-handling controls so any answer can be reconstructed and reviewed.",
        ],
    )

    add_page_break(doc)


def build_llm_provider_portability(doc: Document) -> None:
    """Section 10 - what changes if the team needs to move off Bedrock + Anthropic."""
    add_heading(doc, "10. LLM Provider Portability", level=1)

    add_paragraph(
        doc,
        "The system is intentionally pinned to AWS Bedrock and Anthropic Claude "
        "Sonnet 4.5 today, but the design treats the foundation-model layer as "
        "a swappable component. This section describes how a future team would "
        "migrate the runtime onto a different provider with minimal disruption "
        "and explains why the abstraction is small enough to make that migration "
        "a focused exercise rather than a rewrite.",
    )

    add_heading(doc, "10.1 The Single Choke Point", level=2)
    add_paragraph(
        doc,
        "All provider interactions flow through one module - "
        "app/services/bedrock_provider.py - which exposes exactly two methods:",
    )
    add_bullets(
        doc,
        [
            "converse(system_prompt, user_prompt, max_tokens=...) returns a result with .text, .input_tokens, .output_tokens.",
            "embed_text(text) returns a Python list of floats (1024-dim, unit-norm by default).",
        ],
    )
    add_paragraph(
        doc,
        "Every other module - the bias detector, the retriever, the ingestion "
        "pipeline - calls these two methods. Switching providers means writing "
        "an alternative provider class with the same two-method shape; nothing "
        "else needs to change.",
    )

    add_heading(doc, "10.2 What Stays Stable Across Providers", level=2)
    add_bullets(
        doc,
        [
            "The system-prompt + user-prompt structure (most modern providers accept this shape natively).",
            "The strict JSON schema for output - it is defined in prompts/bias_detection_system_prompt.txt, not in the provider.",
            "The retrieval logic - FAISS, MMR, and concept filtering are provider-independent.",
            "The taxonomy and the response payload - both are provider-independent.",
            "The vector dimension is recorded in vector_store/manifest.json; if the new provider's embedding model has a different native dimension, the corpus is rebuilt and the manifest is updated.",
        ],
    )

    add_heading(doc, "10.3 What Changes Per Provider", level=2)
    add_header_table(
        doc,
        ["Provider", "Auth", "Chat model", "Embedding model", "Notes"],
        [
            [
                "Anthropic API (direct)",
                "ANTHROPIC_API_KEY",
                "claude-sonnet-4-5 (latest)",
                "Anthropic does not host an embedding model; pair with Voyage AI or OpenAI",
                "Smallest delta from current state - same model family, different transport",
            ],
            [
                "OpenAI",
                "OPENAI_API_KEY",
                "gpt-5 / latest reasoning model",
                "text-embedding-3-large (3072) or -small (1536)",
                "Confirm strict JSON via response_format=json_object; corpus rebuild required for new dim",
            ],
            [
                "Google Vertex AI",
                "GOOGLE_APPLICATION_CREDENTIALS",
                "Gemini 2.5 Pro / latest",
                "text-embedding-004 (768) or text-embedding-large-002",
                "Use Vertex SDK; structured-output mode supported; corpus rebuild for new dim",
            ],
            [
                "Azure OpenAI",
                "AZURE_OPENAI_KEY + endpoint",
                "Same OpenAI models behind an Azure deployment",
                "Same OpenAI embeddings",
                "Practically identical to OpenAI; differs only in auth and endpoint URL",
            ],
            [
                "Self-hosted (Llama / Mistral)",
                "Local TLS or VPC-private",
                "Llama-3.1-70B-Instruct or Mistral-Large-Instruct",
                "BAAI/bge-large-en-v1.5 or e5-mistral-7b-instruct",
                "Add a vLLM / TGI front end; structured output via grammar constraints; corpus rebuild for new dim",
            ],
        ],
    )

    add_heading(doc, "10.4 Migration Steps", level=2)
    add_paragraph(doc, "Switching providers is a four-step exercise:")
    add_numbered(
        doc,
        [
            "Implement a new provider class (e.g. OpenAIProvider, GeminiProvider) in app/services/ with the same converse() and embed_text() signature as BedrockProvider.",
            "Add a small factory in shared_components/settings.py that returns the configured provider (LLM_PROVIDER env var) - default remains 'bedrock'.",
            "Re-embed every corpus with the new embedding model (python data_pipeline/build_vector_index.py --corpus public/private). The vector dimension is whatever the new provider returns; it is recorded in manifest.json.",
            "Re-run the evaluation harness and scripts/compare_versions.py against the new provider; document the delta in the next release note before promoting traffic.",
        ],
    )

    add_heading(doc, "10.5 Cross-Cutting Concerns to Re-Validate Per Provider", level=2)
    add_header_table(
        doc,
        ["Concern", "Why it matters", "How to validate"],
        [
            ["Strict JSON output", "The bias detector relies on a closed schema", "Run evaluation harness; expect 100 percent schema fidelity"],
            ["Closed-taxonomy adherence", "The model must not invent bias names", "Validate every bias_name in responses against bias-taxonomy.json"],
            ["Determinism", "Comparison runs need stable output", "Run a scenario five times; flag if results vary non-trivially at temperature 0"],
            ["Long-output reliability", "Schema-rich JSON can be 2-3 KB", "Check max_tokens setting and confirm no truncation in evaluation"],
            ["Embedding semantics", "Different models cluster meaning differently", "Manually inspect retrieval for canonical scenarios; expect concept drift but not collapse"],
            ["Cost and latency", "New provider may have different SLOs", "Re-baseline P50/P95 metrics in the dev environment before production cut-over"],
            ["Data residency and contractual data use", "Provider terms differ", "Confirm the provider does not use prompts for training; verify regional availability"],
        ],
    )

    add_heading(doc, "10.6 Why the Default Stays Bedrock", level=2)
    add_bullets(
        doc,
        [
            "One control plane for chat and embeddings - same boto3 client, same IAM identity, same VPC pattern.",
            "Enterprise posture - VPC endpoints, KMS encryption, CloudTrail audit, no model-vendor sprawl.",
            "Bedrock does not use prompts to train foundation models, which matters for any decision-support context that may handle sensitive scenarios.",
            "Aligned with the AWS-native stack the project assumes for hosting and ML-Ops (S3 for vector store, CloudWatch for observability, IAM for auth).",
        ],
    )

    add_paragraph(
        doc,
        "The provider-portability work above is not speculative architecture - "
        "it is the natural consequence of building the system around a small "
        "interface (converse + embed_text) and treating both prompts and "
        "evaluation as artefacts owned by the project rather than the vendor. "
        "If a future commercial, regulatory, or pricing change forces a move, "
        "the design above is what makes that move a focused engineering "
        "project rather than a rewrite.",
    )

    add_page_break(doc)


def build_risks(doc: Document) -> None:
    """Risks, limitations, and mitigations."""
    add_heading(doc, "11. Risks, Limitations, and Mitigations", level=1)

    add_header_table(
        doc,
        ["Risk", "Impact", "Mitigation"],
        [
            ["LLM hallucination of bias names", "Decision support degraded by invalid taxonomy entries", "Closed-taxonomy injection in system prompt; post-hoc validation against taxonomy keys"],
            ["Embedding-paraphrase drift", "Retrieval may surface only loosely relevant chunks", "Concept and decision-domain filters on top-k * 3 candidates; expand corpus and retune top-k"],
            ["Model upgrade behaviour change", "Output schema or quality regresses silently", "Pinned model id in env; evaluation harness as deployment gate"],
            ["Bedrock service disruption", "Runtime path fails", "Surface ``BedrockInferenceError``; future: regional failover and graceful baseline-only mode"],
            ["Sensitive-content leakage in logs", "Privacy breach", "Default redact scenarios in CloudWatch; retention policy and KMS-encrypted log groups"],
            ["Misuse as authoritative advice", "Harm to users in legal/HR/medical contexts", "Explicit disclaimer surfaced in UI and in every output payload; training material for stakeholders"],
            ["Corpus bias toward English/Western literature", "Recommendations skew culturally", "Document this limitation; expand corpus across languages and regions"],
        ],
    )

    add_page_break(doc)


def build_roadmap(doc: Document) -> None:
    """Roadmap and next steps."""
    add_heading(doc, "12. Roadmap and Next Steps", level=1)

    add_heading(doc, "12.1 Short Term", level=2)
    add_bullets(
        doc,
        [
            "Validate the live Bedrock embedding and chat paths end-to-end against a clean rebuild.",
            "Tune ``top_k`` and concept/domain filters using inspection runs over the existing scenarios.",
            "Add automated schema and taxonomy adherence checks to the evaluation runner.",
            "Implement a minimal Streamlit or FastAPI demo surface for recruiter walk-throughs.",
        ],
    )

    add_heading(doc, "12.2 Medium Term", level=2)
    add_bullets(
        doc,
        [
            "Expand corpus into Phase 2 (Just Mercy, Groupthink, Mistakes Were Made, Invisible Women, Sources of Power) and Phase 3 (Behave, Human Compatible, Atlas of AI, The Black Swan).",
            "Add LLM-as-judge grounding evaluations and a reviewer rubric for usefulness/clarity.",
            "Implement S3-backed vector store hot-reload so corpus refreshes do not require a redeploy.",
            "Containerise the runtime and stand up the AWS Fargate deployment with a Bedrock VPC endpoint.",
        ],
    )

    add_heading(doc, "12.3 Long Term", level=2)
    add_bullets(
        doc,
        [
            "Multi-tenant capability with per-tenant taxonomy overlays and audit logs.",
            "Optional fine-tuned reranker over the FAISS candidate set for higher-precision retrieval.",
            "Periodic ANN index migration (IndexIVFFlat / HNSW) when corpus passes ~50k chunks.",
            "Integration with structured assessment tools (interview scoring, case-management workflows) as a sidecar.",
        ],
    )

    add_page_break(doc)


def build_appendix(doc: Document) -> None:
    """Glossary and references."""
    add_heading(doc, "13. Appendix", level=1)

    add_heading(doc, "13.1 Glossary", level=2)
    add_definition_table(
        doc,
        [
            ("RAG", "Retrieval-Augmented Generation. Pattern that enriches an LLM prompt with retrieved passages."),
            ("Embedding", "A fixed-length numeric vector that represents the meaning of a piece of text."),
            ("FAISS", "Facebook AI Similarity Search - a high-performance library for similarity search over dense vectors."),
            ("IndexFlatIP", "FAISS index type that performs exhaustive inner-product search; equivalent to cosine similarity on unit vectors."),
            ("Bedrock", "AWS managed service that exposes foundation models (Claude, Titan, Llama, etc.) through a single API."),
            ("Titan v2", "``amazon.titan-embed-text-v2:0`` - Amazon's hosted text embedding model used here at 1024 dimensions."),
            ("Claude Sonnet 4.5", "``us.anthropic.claude-sonnet-4-5-20250929-v1:0`` - the chat model used for bias detection and reasoning."),
            ("Concept catalog", "Unified dictionary built from ``bias-taxonomy.json`` and ``retrieval-concepts.json`` and used for tagging and filtering."),
            ("Cleaning notes", "Per-chunk audit flags recorded when front- or back-matter was trimmed."),
            ("Parser notes", "Per-document audit flags recorded when a metadata field could not be inferred."),
            ("Baseline path", "LLM call without retrieved context - used as the comparison baseline."),
            ("RAG path", "LLM call with retrieved context block injected into the system prompt."),
            ("Noise (Kahneman et al.)", "Inconsistency in judgment between reviewers or across occasions, distinct from bias."),
        ],
    )

    add_heading(doc, "13.2 Primary Source Books in the Corpus", level=2)
    add_bullets(
        doc,
        [
            "Thinking, Fast and Slow - Daniel Kahneman",
            "Noise: A Flaw in Human Judgment - Kahneman, Sibony, Sunstein",
            "Judgment Under Uncertainty: Heuristics and Biases - Kahneman, Slovic, Tversky (eds.)",
            "Superforecasting - Tetlock and Gardner",
            "Influence - Cialdini",
            "Weapons of Math Destruction - Cathy O'Neil",
            "Nudge - Thaler and Sunstein",
            "Just Mercy - Bryan Stevenson",
            "Groupthink - Irving Janis",
            "Mistakes Were Made (But Not by Me) - Tavris and Aronson",
            "Invisible Women - Caroline Criado Perez",
            "Sources of Power - Gary Klein",
        ],
    )

    add_heading(doc, "13.3 References", level=2)
    add_bullets(
        doc,
        [
            "AWS - Amazon Bedrock developer documentation: https://docs.aws.amazon.com/bedrock/",
            "Amazon Titan Text Embeddings v2 model card: https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html",
            "Anthropic Claude on Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic.html",
            "FAISS - Facebook AI Similarity Search: https://github.com/facebookresearch/faiss",
            "PyMuPDF documentation: https://pymupdf.readthedocs.io/",
            "AWS Well-Architected Framework: https://aws.amazon.com/architecture/well-architected/",
        ],
    )


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

def configure_footer(doc: Document) -> None:
    """Add a centred page-number footer that auto-updates across the document."""
    section = doc.sections[0]
    footer = section.footer
    footer_para = footer.paragraphs[0]
    footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer_para.add_run("Solution Design Document - Bias-Aware Decision Making RAG Copilot - Page ")
    run.font.size = Pt(9)
    run.font.color.rgb = MID_GREY
    _add_page_number_field(footer_para)


def main() -> None:
    """Build the .docx file and write it to ``docs/Solution Design Document.docx``."""
    doc = Document()
    configure_styles(doc)
    configure_footer(doc)

    build_cover_page(doc)
    build_document_control(doc)
    build_executive_summary(doc)
    build_background(doc)
    build_objectives(doc)
    build_solution_overview(doc)
    build_detailed_design(doc)
    build_key_concepts(doc)
    build_validation_qa(doc)
    build_infrastructure(doc)
    build_mlops(doc)
    build_llm_provider_portability(doc)
    build_risks(doc)
    build_roadmap(doc)
    build_appendix(doc)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
