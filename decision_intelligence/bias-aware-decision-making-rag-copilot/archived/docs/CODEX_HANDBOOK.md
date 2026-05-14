# Codex Handoff — Applied-AI-for-Societal-Impact

## Project purpose
Build a portfolio-grade, enterprise-style AI system that helps decision makers reason more carefully about complex human situations using a bias-aware RAG copilot.

The current flagship project is:
`decision_intelligence/bias-aware-decision-making-rag-copilot`

## What the system does
The copilot should:
- accept a scenario involving people, judgments, or ambiguous decisions
- detect likely cognitive, organizational, and AI/system biases
- retrieve supporting context from a curated book-based knowledge base
- return a structured recommendation with caveats, tradeoffs, and mitigation steps
- compare a baseline LLM response vs a RAG-enhanced response

## Content sources
Primary source material includes scholarly and seminal works on decision-making bias and noise, especially:
- Noise: A Flaw in Human Judgment
- Thinking, Fast and Slow
- other related books on bias, judgment, heuristics, and decision hygiene

## Important disclaimer
The system is not legal advice, medical advice, or HR policy enforcement. It is intended to present research-grounded perspectives and encourage better decision making.

## Current state of the repo
The project has already been moving through these stages:
1. PDF ingestion from `data/corpora/<corpus>/raw/` (public or private)
2. text parsing to `data/corpora/<corpus>/parsed_text/`
3. chunk generation to `data/corpora/<corpus>/chunks/chunks.json` (profile-aware)
4. concept extraction to `data/corpora/<corpus>/chunks/chunks_with_concepts.json`
5. enrichment to `data/corpora/<corpus>/knowledge/knowledge_base.json` (passage_type + decision_phase + chapter_title, summaries, keywords, …)
6. **vector index:** `data_pipeline/build_vector_index.py` — Titan v2 embeddings with **sentence-centred ±`RAG_EMBED_SENTENCE_RADIUS` windows**, mean-pooled per chunk (configurable / disable via `RAG_EMBED_*` env vars); FAISS `IndexFlatIP`; `manifest.json` records embedding settings
7. **runtime RAG:** `rag/retriever.py` + `app/services/bias_detector.py` (baseline vs RAG, strict JSON)
8. **evaluation spine:** `evaluation/connectivity.py`, `metrics.py`, `run_card.py`, `scripts/record_response_run.py`, `response/` captures

## Key implementation goals
Codex should continue by making the pipeline robust and traceable:
- extract author metadata from the first pages of each PDF
- keep source file, author, and book title attached to every chunk
- preserve an auditable chain from PDF -> parsed text -> chunk -> concept tags -> enriched knowledge
- make the knowledge base retrieval-ready for FAISS or a similar vector store
- add a comparison mode: baseline LLM vs RAG-enhanced LLM (**implemented** in `bias_detector.py`; frozen scenarios + `scripts/record_response_run.py` persist captures under `response/`)

## Repo layout guidance
Use this structure inside the project root:

```text
bias-aware-decision-making-rag-copilot/
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── parsed_text/
│   │   ├── chunks/
│   │   └── knowledge/
│   └── metadata/
├── data_pipeline/
├── shared_components/
│   └── utilities/
├── rag/
├── prompts/
├── notebooks/
└── README.md
```

## Python packaging guidance
Use a shared bootstrap utility to add the project root to `sys.path` when scripts are run directly.

Recommended pattern:
- `data_pipeline/bootstrap.py`
- `shared_components/utilities/path_utils.py`
- `__init__.py` files in package directories

## Run order
A clean run sequence should be:
1. parse PDFs
2. chunk parsed text
3. extract concepts
4. enrich chunks
5. build vector index (`build_vector_index.py`: sentence-window Titan pooling per chunk → `embeddings.npy` + FAISS)
6. run RAG retrieval (`rag/retriever.py`)
7. compare baseline vs RAG outputs (`bias_detector.py`, `scripts/record_response_run.py`, etc.)

## What Codex should do next
1. finish the metadata-aware ingestion pipeline
2. implement embeddings and retrieval
3. add a retriever and a RAG orchestration layer
4. add evaluation utilities for retrieval quality and response quality
5. add a simple UI or API layer for demo use

## Style rules for the codebase
- keep files small and single-purpose
- prefer explicit, well-documented functions
- keep path logic centralized
- preserve traceability and explainability in all outputs
- avoid hardcoding secrets
- keep prompt files and taxonomy files outside code when practical
