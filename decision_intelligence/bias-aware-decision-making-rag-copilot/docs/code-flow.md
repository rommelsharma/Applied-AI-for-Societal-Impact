# Code Flow

This document describes the repository layout and the end-to-end execution flow of the Bias-Aware Decision Making RAG Copilot. It is intended to be the single navigational reference for new contributors, code reviewers, and recruiters auditing the system.

---

## 1. Top-Level Directory Structure

```text
bias-aware-decision-making-rag-copilot/
│
├── app/                      # Application services (LLM provider, bias detector)
│   └── services/
│       ├── bedrock_provider.py
│       └── bias_detector.py
│
├── rag/                      # Retrieval layer
│   └── retriever.py
│
├── data_pipeline/            # Offline ingestion + index build pipeline
│   ├── bootstrap.py
│   ├── pdf_parser.py
│   ├── chunker.py
│   ├── concept_extractor.py
│   ├── enrich_chunks.py
│   ├── build_knowledge_base.py
│   └── build_vector_index.py
│
├── shared_components/        # Cross-cutting utilities and configuration
│   ├── settings.py
│   └── utilities/
│       ├── path_utils.py
│       └── taxonomy_utils.py
│
├── prompts/                  # LLM prompt templates
│   └── bias_detection_system_prompt.txt
│
├── data/
│   ├── corpora/                      # All corpus content lives here
│   │   ├── public/                   # Synthesised dossiers (committed)
│   │   │   ├── raw/                  # Source PDFs (committed)
│   │   │   ├── parsed_text/          # Extracted text + metadata sidecars (gitignored)
│   │   │   ├── chunks/               # Chunked + concept-tagged JSON (gitignored)
│   │   │   ├── knowledge/            # Final enriched knowledge base (gitignored)
│   │   │   └── vector_store/         # Embeddings + FAISS index + metadata (gitignored)
│   │   └── private/                  # Full-book PDFs (entire dir gitignored, local-only)
│   │       └── (same five subfolders, all gitignored)
│   └── metadata/                     # Corpus-independent: shared across corpora
│       ├── bias-taxonomy.json
│       ├── retrieval-concepts.json
│       └── book_dossiers/            # Source manifests + generated HTML/MD/PDF
│
├── evaluation/               # Test scenarios and evaluation runner
│   ├── test_scenarios.json
│   └── run_evaluation.py
│
├── notebooks/                # Interactive demo entry point
│   └── test_bias_detection.py
│
├── scripts/                  # Auxiliary utilities (dossier generation, HTML→PDF)
│   ├── generate_book_dossiers.py
│   └── render_html_to_pdf.mjs
│
├── docs/                     # Architecture, project, and onboarding docs
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 2. Two Execution Phases

The system has two distinct execution paths:

1. **Offline pipeline** — runs once (or on corpus refresh) to produce the searchable knowledge base.
2. **Runtime path** — runs every time a user submits a scenario; it retrieves and reasons.

Both paths share configuration (`shared_components/settings.py`), path resolution (`shared_components/utilities/path_utils.py`), taxonomy loading (`shared_components/utilities/taxonomy_utils.py`), and the Bedrock provider (`app/services/bedrock_provider.py`).

---

## 3. Offline Ingestion Pipeline

The offline pipeline transforms raw PDFs into a vector-searchable, traceable knowledge base.

### Logical step sequence

```text
PDFs (data/corpora/<corpus>/raw/)
   │
   ▼
[1] pdf_parser.py        →  data/corpora/<corpus>/parsed_text/{slug}.txt + .meta.json
   │
   ▼
[2] chunker.py           →  data/corpora/<corpus>/chunks/chunks.json
   │                       (profile-aware: dossier 350w / book 500w / auto)
   ▼
[3] concept_extractor.py →  data/corpora/<corpus>/chunks/chunks_with_concepts.json
   │
   ▼
[4] enrich_chunks.py     →  data/corpora/<corpus>/knowledge/knowledge_base.json
   │                       (adds passage_type, decision_phase, chapter_title)
   ▼
[5] build_vector_index.py → data/corpora/<corpus>/vector_store/{embeddings.npy, knowledge.index, index_metadata.json, manifest.json}
```

`data_pipeline/build_knowledge_base.py` orchestrates steps 1–4 in one entry point. Step 5 is a separate script because it requires live Bedrock access and produces large binary artifacts.

### File-by-file description

| File | Responsibility |
|------|---------------|
| `data_pipeline/bootstrap.py` | Adds the project root to `sys.path` so scripts can be run directly from PyCharm, terminal, or `python -m …` without installation. |
| `data_pipeline/pdf_parser.py` | Reads every PDF in `<corpus>/raw/`, extracts text via PyMuPDF, joins lines that were soft-broken by hyphenation, infers author / title / publication year from the first pages with regex heuristics, and writes a paired `.txt` plus `.meta.json` sidecar per book. Records `parser_notes` when a field could not be inferred. Corpus-aware via `--corpus` and `CORPUS_NAME`. |
| `data_pipeline/chunker.py` | Loads each parsed text plus its metadata sidecar, trims an expanded list of front- and back-matter markers (preface, foreword, prologue, acknowledgements, notes, bibliography, index, glossary, appendix, "about the author", "also by …"), normalises whitespace, and splits the body using one of three profiles: `dossier` (350 / 60), `book` (500 / 80), or `auto` (`book` when the document exceeds ~30 k words, otherwise `dossier`). Detects chapter boundaries and stamps `chapter_title` onto every chunk. |
| `data_pipeline/concept_extractor.py` | Builds a unified concept catalog from `bias-taxonomy.json` and `retrieval-concepts.json`, then tags every chunk with the concepts whose canonical name or aliases appear in the text. Emits `concepts` + `concept_confidence` (medium for one alias hit, high for two or more). |
| `data_pipeline/passage_classifier.py` | Lightweight per-chunk classifier producing `passage_type` (prescription / framework / study / case_study / anecdote / definition) and `decision_phase` (diagnose / design / decide / review). Heuristic by default; routes through Claude Haiku when `ENRICHMENT_USE_LLM=true`, with automatic fallback to the rule-based path on failure. |
| `data_pipeline/enrich_chunks.py` | Final pre-embedding enrichment: generates a short summary, derives `importance`, `decision_domains`, top keywords, and the `passage_type` / `decision_phase` tags from the classifier above. Writes `<corpus>/knowledge/knowledge_base.json` with every traceability field preserved (parser notes, cleaning notes, chapter title, chunking profile). |
| `data_pipeline/build_knowledge_base.py` | Orchestrator that runs parse → chunk → extract concepts → enrich in sequence with progress banners. Supports `--corpus` and `--profile`. |
| `data_pipeline/build_vector_index.py` | Reads `knowledge_base.json`, calls Bedrock Titan v2 (`amazon.titan-embed-text-v2:0`) for each chunk, stacks the vectors into a `float32` matrix, persists `embeddings.npy`, builds a FAISS `IndexFlatIP` (cosine-equivalent on normalised embeddings), and writes `index_metadata.json` plus `manifest.json` (corpus name, chunk count, dimension, index type, profile distribution). `--metadata-only` refreshes only the alignment metadata when chunk text is unchanged - useful after extending enrichment without re-embedding. |

### Cross-cutting helpers used by the pipeline

| File | Responsibility |
|------|---------------|
| `shared_components/settings.py` | Loads `.env` and exposes two frozen dataclasses: `BEDROCK_SETTINGS` (region, chat / embedding model ids, endpoint URL, embedding dimensions) and `RAG_SETTINGS` (corpus name, chunking profile, default `top_k`, MMR lambda, reranker config, LLM-enrichment toggle). Single source of truth for configuration. |
| `shared_components/utilities/path_utils.py` | Corpus-aware path resolution for `data/corpora/<corpus>/{raw,parsed_text,chunks,knowledge,vector_store}`, plus the corpus-independent `data/metadata`, `prompts/`, `evaluation/`. Optional `corpus` argument lets a single process address multiple corpora simultaneously (used by `compare_versions.py`). |
| `shared_components/utilities/taxonomy_utils.py` | Lazy-cached loaders for `bias-taxonomy.json` and `retrieval-concepts.json`, plus a unified `build_concept_catalog()` that merges both into a single dict keyed by concept name. |

---

## 4. Runtime Path: Scenario → Bias Analysis

This is the user-facing flow invoked when a scenario is submitted (via `evaluation/run_evaluation.py`, `notebooks/test_bias_detection.py`, or any future API/UI surface).

### Logical step sequence

```text
User scenario (string)
   │
   ▼
[A] concept_extractor.extract_concepts(scenario)
       │  (taxonomy-driven concept tagging on the user's text)
   ▼
[B] KnowledgeRetriever.search(scenario, preferred_concepts=…)
       │  • embed scenario via Bedrock Titan v2
       │  • FAISS IP search over knowledge_base vectors (top_k * 3 or reranker_candidates)
       │  • optional concept / domain filtering
       │  • optional Claude-Haiku reranker (off by default)
       │  • MMR diversification across remaining candidates
       │  • unfiltered fallback if filters yield nothing
   ▼
[C] format_retrieved_context(results)
       │  → numbered context blocks with source, author, concepts, summary, excerpt
   ▼
[D] build_system_prompt(base_prompt, taxonomy, rag_context=None)   ← baseline
[D] build_system_prompt(base_prompt, taxonomy, rag_context=ctx)    ← RAG
   ▼
[E] BedrockProvider.converse(...) called twice
       │  Claude Sonnet 4.5 returns strict JSON for both prompts
       │  Malformed JSON triggers one stricter retry
   ▼
[F] {"without_rag": …, "with_rag": …, "retrieval": [traceability rows]}
```

### File-by-file description

| File | Responsibility |
|------|---------------|
| `app/services/bedrock_provider.py` | Thin Bedrock client wrapper. Constructs `bedrock-runtime` boto3 client using region + optional endpoint URL, exposes `converse()` for chat completion against Claude Sonnet 4.5 and `embed_text()` for Titan v2 embeddings. Surfaces `BedrockConfigurationError` and `BedrockInferenceError` for clean upstream handling. |
| `rag/retriever.py` | Corpus-aware retriever. Loads the FAISS index (or numpy fallback) plus the aligned metadata for the chosen corpus at construction time. `search()` embeds the query, pulls a wide candidate pool, applies concept / decision-domain filters with an unfiltered fallback, optionally reranks the candidates with Claude Haiku as a relevance judge, and diversifies the final cut with Maximal Marginal Relevance so a single book or chapter cannot monopolise the result set. Returns `RetrievalResult` dataclasses with chapter, passage type, decision phase, and similarity score. |
| `app/services/bias_detector.py` | The orchestration brain. Loads the taxonomy and prompt template, extracts concepts from the user scenario, runs retrieval, formats context, builds two system prompts (no-RAG baseline and RAG-enhanced), invokes Claude twice via the provider, parses strict JSON (with one fenced-code-aware retry), and returns a comparison dict ready for evaluation or UI rendering. |
| `prompts/bias_detection_system_prompt.txt` | The locked output contract for the LLM: role, allowed taxonomy use, JSON schema, quality expectations, and a worked example. Kept outside code so prompt iteration does not require code changes. |

---

## 5. Evaluation and Demo Surfaces

| File | Responsibility |
|------|---------------|
| `evaluation/test_scenarios.json` | ~30 realistic decision-making scenarios spanning hiring, performance review, sunk cost, automation bias, groupthink, base-rate neglect, AI alignment, sentencing, narrative fallacy, instrumental convergence, and more. |
| `evaluation/run_evaluation.py` | Iterates through the scenarios, calls `detect_bias_comparison`, prints lightweight diff stats, and persists `latest_results.json`. Honours `CORPUS_NAME` so it can be aimed at any corpus. |
| `notebooks/test_bias_detection.py` | Interactive walkthrough of the same scenarios, designed to run in PyCharm or a notebook for visual inspection of baseline vs RAG output. |
| `scripts/run_sample_comparison.py` | Runs two illustrative scenarios end-to-end and writes both raw JSON and a reviewer-facing markdown report under `evaluation/` and `docs/`. |
| `scripts/compare_versions.py` | Runs the full scenario suite (or any subset) against multiple corpus versions side-by-side. Emits both a JSON sidecar (full payloads, kept locally) and a markdown summary (counts and named biases only - safe to share). |
| `scripts/generate_book_dossiers.py` | Generates retrieval-friendly book dossiers (Markdown + HTML + PDF) from a JSON manifest in `data/metadata/book_dossiers/`. The PDFs land in `data/corpora/public/raw/` and feed the offline pipeline. |
| `scripts/render_html_to_pdf.mjs` | Optional Playwright-based pipeline that renders the generated HTML to high-fidelity PDF when the ReportLab path is not preferred. |

---

## 6. Configuration and Secrets

`.env` (not committed) supplies:

```text
AWS_BEARER_TOKEN_BEDROCK=…
BEDROCK_REGION=us-west-2
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_ENDPOINT_URL=https://bedrock-runtime.us-west-2.amazonaws.com/
BEDROCK_EMBEDDING_DIMENSIONS=1024

CORPUS_NAME=public           # 'public' (committed dossiers) or 'private' (full books, local-only)
CHUNK_PROFILE=auto           # dossier | book | auto
RAG_TOP_K=8
RAG_MMR_LAMBDA=0.7           # 1.0 = pure relevance, 0.0 = pure diversity
RAG_RERANKER=none            # 'none' | 'claude_haiku'
RAG_RERANKER_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0
RAG_RERANKER_CANDIDATES=24
ENRICHMENT_USE_LLM=false     # rule-based passage classifier when false
```

`shared_components/settings.py` is the only place that reads these values. `BedrockProvider.__init__` validates that the bearer token is present before building a boto3 client.

---

## 7. Run Order Cheat-Sheet

Public-corpus build (committed dossiers, default):

```bash
python data_pipeline/build_knowledge_base.py --corpus public      # parse → chunk → concepts → enrich
python data_pipeline/build_vector_index.py --corpus public        # embed + FAISS index
```

Private-corpus build (full books on your local laptop only):

```bash
python data_pipeline/build_knowledge_base.py --corpus private --profile book
python data_pipeline/build_vector_index.py --corpus private
```

Run a scenario interactively (defaults to public; honours `CORPUS_NAME`):

```bash
python notebooks/test_bias_detection.py
CORPUS_NAME=private python notebooks/test_bias_detection.py
```

Run the evaluation suite:

```bash
python evaluation/run_evaluation.py
```

Compare two corpus versions side-by-side:

```bash
python scripts/compare_versions.py --versions public,private
python scripts/compare_versions.py --versions public,private --scenario-ids 11,12
```

The output of `compare_versions.py` is two files in `evaluation/`: a JSON
sidecar (full payloads, intended for local analysis) and a markdown
summary (counts and bias names only - safe to share publicly).

---

## 8. Public vs Private Corpus

This project ships with two parallel corpora:

| Corpus | What goes in | Where it lives | Committed? |
|---|---|---|---|
| `public` | Synthesised, retrieval-friendly book dossiers | `data/corpora/public/` | Raw PDFs committed; derivatives gitignored |
| `private` | Full-book PDFs (personal research / private demo only) | `data/corpora/private/` | **Never committed** - directory fully gitignored |

The split exists so the demo and any deployed runtime stay safe to ship,
while the private corpus serves as an internal evaluation harness that
shows the *actual* lift the system reaches with real source material.
The recommended convention is to publish public-corpus numbers in any
shared report and use `compare_versions.py` to honestly disclose the
delta between corpora when relevant.

See [`PRIVATE_CORPUS_GUIDE.md`](PRIVATE_CORPUS_GUIDE.md) for the full
private-corpus runbook.

---

## 9. Traceability Contract

Every artefact preserves the chain from PDF to LLM output:

- Each chunk carries `id`, `source`, `file_name`, `author`, `title`, `publication_year`, `chunk_index`, `cleaning_notes`, `parser_notes`, `chapter_title`, `chunking_profile`, `passage_type`, and `decision_phase`.
- Each retrieval result carries those fields plus the similarity `score` and matched `concepts`.
- The bias detector's response payload includes a `corpus` field plus a `retrieval` array so the calling layer can show *which* chunks - and from *which* corpus - shaped the answer.

This contract is what allows the system to satisfy the explainability and source-attribution principles stated in `docs/PROJECT_CONTEXT.md`.
