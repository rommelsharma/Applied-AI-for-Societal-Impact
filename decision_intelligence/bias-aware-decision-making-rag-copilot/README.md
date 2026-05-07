# Bias-Aware Decision Making RAG Copilot

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) based AI assistant** designed to support better decision-making by identifying cognitive biases and reducing inconsistency (“noise”) in human judgment.

The system is grounded in behavioral science research, referring to focused work on the subject, for example:

* *Noise: A Flaw in Human Judgment* — Daniel Kahneman, Olivier Sibony, Cass Sunstein
* *Thinking, Fast and Slow* — Daniel Kahneman

---

## Objective

To build a practical AI system that:

* Helps users analyze complex situations involving multiple stakeholders
* Identifies potential cognitive biases and judgment noise
* Provides structured, research-backed perspectives to improve decisions

---

## Intended Use

This tool is designed for individuals involved in decision-making, including:

* Leaders and managers
* HR professionals
* Legal and compliance contexts
* Teams handling complex interpersonal situations
* Individuals making high-impact personal or professional decisions

---

## Disclaimer

This system is **not a source of legal, financial, or medical advice**.

It provides:

* Research-informed perspectives
* Structured thinking guidance

It does **not replace professional judgment** or domain expertise.

---

## How It Works

1. User provides a scenario or decision context
2. System retrieves relevant insights from curated literature
3. LLM generates a structured response highlighting:

   * Potential biases
   * Noise in judgment
   * Risks and considerations
   * Suggested approaches

---

## Core Components

* Document ingestion and preprocessing
* Metadata-aware bias taxonomy and retrieval concept catalog
* Bedrock-powered embeddings and vector search
* Retrieval pipeline (RAG)
* LLM-based response generation
* Evaluation and comparison flow

---

## Project Structure

```id="5n9kq2"
bias-aware-decision-making-rag-copilot/
│
├── app/                # Backend orchestration (bias detector, Bedrock provider)
├── rag/                # FAISS retriever with MMR + optional reranker
├── data_pipeline/      # PDF parser → chunker → enricher → vector index
├── shared_components/  # Settings + path utilities + taxonomy loaders
├── data/
│   ├── corpora/
│   │   ├── public/     # Synthesised dossiers (raw PDFs committed)
│   │   └── private/    # Full books (entirely local-only, gitignored)
│   └── metadata/       # Bias taxonomy + retrieval concepts (shared)
├── prompts/            # Prompt templates
├── evaluation/         # Test scenarios + scorecards + run output
├── scripts/            # Dossier generation, sample comparison, version comparison
├── notebooks/          # Interactive demo entry point
├── docs/               # code-flow, fundamental concepts, private corpus guide, etc.
└── README.md
```

Two parallel corpora are supported:

* **`public`** - synthesised, retrieval-friendly dossiers. Raw PDFs are committed; derived artefacts can be regenerated and are gitignored.
* **`private`** - full-book PDFs you own personally for research / private demo. The whole `data/corpora/private/` tree is gitignored and never committed.

See [`docs/PRIVATE_CORPUS_GUIDE.md`](docs/PRIVATE_CORPUS_GUIDE.md) for the
private-corpus runbook and [`docs/code-flow.md`](docs/code-flow.md) for
the full architecture walkthrough.

---

## Expected Output (Example Structure)

* Situation Summary
* Identified Biases
* Noise Considerations
* Risks
* Suggested Actions
* Supporting Insights

---

## Status

Foundation build in progress.

Implemented now:

* PDF parsing with metadata sidecars + hyphenation cleanup
* Front/back-matter trimming with expanded marker set
* Profile-aware chunking (`dossier` 350 / 60, `book` 500 / 80, `auto`)
* Chapter-aware metadata (`chapter_title`)
* Taxonomy-driven concept tagging + bias taxonomy of 60+ entries (cognitive, systemic, AI alignment)
* Lightweight `passage_type` / `decision_phase` classifier (rule-based; LLM-upgradable)
* Bedrock provider layer for chat (Claude Sonnet 4.5) and embeddings (Titan v2)
* Corpus-aware ingestion + vector index build (`public` / `private`)
* FAISS `IndexFlatIP` retriever with concept/domain filtering, MMR diversification, and an optional Claude-Haiku reranker hook (off by default)
* Baseline-vs-RAG bias detector returning strict JSON with retrieval traceability
* ~30 evaluation scenarios across hiring, performance review, legal, AI governance, leadership, strategy, and compliance
* `compare_versions.py` for side-by-side public-vs-private corpus evaluation

Still to validate end to end:

* Private-corpus build against full-length book PDFs (run locally)
* Honest A/B numbers on the lift between corpora
* Demo surface for recruiters

---

## Next Steps

* Build the private corpus locally and run `compare_versions.py` to capture honest lift numbers
* Phased rollout (dossiers + author-published essays in public, full books in private eval) — see `docs/PRIVATE_CORPUS_GUIDE.md`
* Promote the rule-based passage classifier to LLM-assisted once the private corpus is in place (`ENRICHMENT_USE_LLM=true`)
* Consider enabling the Claude-Haiku reranker once the corpus exceeds ~3,000 chunks
* Add a simple demo interface

---

## Author

Rommel Sharma
Enterprise Technology Leader | Applied AI Practitioner
LinkedIn: https://www.linkedin.com/in/rommelsharma/
