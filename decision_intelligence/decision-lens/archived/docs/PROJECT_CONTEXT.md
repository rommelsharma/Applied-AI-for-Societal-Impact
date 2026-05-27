# Project Context

## Repository
`Applied-AI-for-Societal-Impact`

## Flagship project
`decision_intelligence/decision-lens`

## Vision
Create a polished enterprise-style AI portfolio project centered on decision intelligence. The project should help users reason about biased or noisy judgments by combining:
- a curated knowledge base from books and scholarly sources
- a bias taxonomy
- a retrieval-augmented generation pipeline
- structured outputs that explain reasoning, tradeoffs, and limits

## Target users
- managers
- leaders
- HR professionals
- legal and policy-adjacent reviewers
- individual contributors making difficult judgment calls
- hiring committees and panels reviewing candidates

## Main user experience
A user submits a complex scenario. The system responds with:
- likely biases and noise sources
- supporting evidence from the knowledge base
- suggested mitigation steps
- a recommendation framed as decision support, not authoritative advice
- a **baseline vs RAG** comparison on every `detect_bias_comparison` call (same JSON schema, retrieval trace included)

## Index and evaluation artefacts
- **Vector index:** FAISS `IndexFlatIP` over Titan v2 vectors. **Chunks** are embedded with **sentence-centred windows** (±3 sentences by default), mean-pooled and normalised—see `data_pipeline/build_vector_index.py` and `docs/fundamental-concepts.md`. **Queries** embed the full scenario string.
- **Frozen scenarios:** canonical `data/eval/gold/scenarios_catalog.json` (baseline slice, extended suite, private-book prompts, gold expectations). Source JSON used to build it lives under `archived/evaluation_jsonsources/` (see `archived/README.md`).
- **Captures:** `scripts/record_response_run.py` writes `data/eval/runs/<timestamp>_<label>_rag_eval.json` with connectivity results, **run cards** (`evaluation/run_card.py`), and **schema/taxonomy metrics** (`evaluation/metrics.py`).

## Principles
- explainability first
- source traceability
- enterprise-grade documentation
- reproducibility
- avoid overstating model certainty
- do not present outputs as legal or professional advice

## Success criteria
The project is successful if it can:
- ingest multiple PDFs reliably
- preserve source metadata
- produce quality chunks and concept tags
- support retrieval over the curated corpus
- generate credible, grounded decision support responses
- clearly demonstrate improvement from RAG over baseline prompting
