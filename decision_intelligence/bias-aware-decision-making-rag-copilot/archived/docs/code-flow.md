# Code Flow

This document describes the repository layout and the end-to-end execution flow of the Bias-Aware Decision Making RAG Copilot. It is intended to be the single navigational reference for new contributors, code reviewers, and HR / Legal / business reviewers auditing the system.

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
│   ├── metadata/                     # Corpus-independent: shared across corpora
│   │   ├── bias-taxonomy.json
│   │   ├── retrieval-concepts.json
│   │   └── book_dossiers/            # Source manifests + generated HTML/MD/PDF
│   └── eval/
│       ├── gold/                     # scenarios_catalog.json (canonical scenarios)
│       └── runs/                     # *_rag_eval.json, connectivity_log, sample_results (gitignored)
│
├── evaluation/               # Harness, metrics, scenario catalog loader
├── response/                 # Legacy placeholder (.gitkeep); captures use data/eval/runs/
├── archived/                 # Legacy JSON sources + old captures (see archived/README.md)
│
├── notebooks/                # Interactive demo entry point
│   └── test_bias_detection.py
│
├── tests/                    # Stdlib unittest (e.g. retriever overlap helpers)
│
├── scripts/                  # Eval capture, sample comparison, dossiers, compare_versions, …
│   ├── record_response_run.py
│   ├── run_sample_comparison.py
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
| `data_pipeline/build_vector_index.py` | Reads `knowledge_base.json`, calls Bedrock Titan v2 (`amazon.titan-embed-text-v2:0`) for each chunk. **Default:** sentence-centred windows (±`RAG_EMBED_SENTENCE_RADIUS` sentences per sentence in the chunk, mean-pooled and L2-normalised) for stronger retrieval context; disable with `RAG_EMBED_SENTENCE_WINDOWS=false` for a single embed of full `text`. Optional `RAG_EMBEDDING_MAX_WINDOWS` subsamples windows to cap Bedrock cost. Stacks vectors into `embeddings.npy`, builds FAISS `IndexFlatIP`, writes `index_metadata.json` and `manifest.json` (includes embedding window settings). `--metadata-only` refreshes metadata when chunk text is unchanged. |

### Cross-cutting helpers used by the pipeline

| File | Responsibility |
|------|---------------|
| `shared_components/settings.py` | Loads `.env` and exposes `BEDROCK_SETTINGS` and `RAG_SETTINGS` (corpus, chunk profile, `top_k`, MMR, reranker, index-time sentence-window flags, **runtime** `overlap_filter`, `overlap_chunk_radius`, pool multipliers, `context_expand_neighbors`, `expand_max_chars_per_side`). |
| `shared_components/utilities/path_utils.py` | Corpus-aware paths under `data/corpora/<corpus>/…`, plus `data/metadata`, `data/eval/…`, `prompts/`, `evaluation/`, and **`data/eval/runs/`** (`get_response_dir()`; legacy `response/` kept for migration). |
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
       │  • FAISS IP search (candidate pool widened when RAG_OVERLAP_FILTER=true)
       │  • optional concept / domain filtering + unfiltered fallback if empty
       │  • optional Claude-Haiku reranker (off by default)
       │  • MMR (larger cut when overlap filter on, for backfill depth)
       │  • optional same-source chunk_index overlap filter + backfill (RAG_OVERLAP_*)
       │  • optional same-source neighbour excerpts on each hit (RAG_CONTEXT_EXPAND_*)
   ▼
[C] format_retrieved_context(results)
       │  → numbered context blocks + optional “Adjacent context” neighbour lines
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
| `app/services/bedrock_provider.py` | Thin Bedrock client wrapper. Constructs `bedrock-runtime` boto3 client using region + optional endpoint URL, exposes `converse()` for chat completion against Claude Sonnet 4.5 and `embed_text()` for Titan v2 embeddings. On successful client construction, appends one **OK** row to `data/eval/runs/connectivity_log.txt` (`evaluation.connectivity.append_connectivity_log_line`) so each materialised client is auditable. Surfaces `BedrockConfigurationError` and `BedrockInferenceError`. |
| `rag/retriever.py` | Corpus-aware retriever. Loads FAISS (or numpy fallback) and `index_metadata.json`; builds a `(source, chunk_index) → row` map for overlap checks and neighbour expansion. `search()` runs filters, optional reranker, MMR, optional **overlap-aware greedy filter** with backfill, then attaches optional **neighbour_blocks** per hit. Returns `RetrievalResult` (chapter, passage type, decision phase, score, `neighbor_blocks`). |
| `app/services/bias_detector.py` | The orchestration brain. Loads the taxonomy and prompt template, extracts concepts from the user scenario, runs retrieval, formats context, builds two system prompts (no-RAG baseline and RAG-enhanced), invokes Claude twice via the provider, parses strict JSON (with one fenced-code-aware retry), and returns a comparison dict ready for evaluation or UI rendering. |
| `prompts/bias_detection_system_prompt.txt` | The locked output contract for the LLM: role, allowed taxonomy use, JSON schema, quality expectations, and a worked example. Kept outside code so prompt iteration does not require code changes. |

---

## 5. Evaluation and Demo Surfaces

| File | Responsibility |
|------|---------------|
| `data/eval/gold/scenarios_catalog.json` | Canonical eval data: baseline freeze, extended suite (~30 scenarios), private-book prompts, and `gold_expectations`. Built via `scripts/build_scenarios_catalog.py` from `archived/evaluation_jsonsources/` when sources are archived. |
| `evaluation/scenario_catalog.py` | Loaders (`get_baseline_scenarios`, `get_extended_test_scenarios`, …) for the catalog above. |
| `evaluation/run_evaluation.py` | Iterates through the extended suite, calls `detect_bias_comparison`, prints lightweight diff stats, and persists `evaluation/latest_results.json`. Honours `CORPUS_NAME` so it can be aimed at any corpus. |
| `evaluation/connectivity.py` | Bedrock + embedding + vector-store + retrieval smoke checks. Every run appends one tab-separated row to `data/eval/runs/connectivity_log.txt` (timestamp, OK/FAIL, detail). Successful `BedrockProvider()` construction also appends its own row from `bedrock_provider.py`. |
| `evaluation/sample_results_log.py` | Append-only JSONL writer for `data/eval/runs/sample_results_comparison.md`; migrates legacy `docs/sample_results_comparison.md` into `data/eval/runs/` once if present. |
| `evaluation/metrics.py` | Schema and taxonomy validation for `without_rag` / `with_rag` payloads plus per-run aggregates. |
| `evaluation/run_card.py` | Reproducibility metadata (timestamps, git commit, file hashes, model/RAG settings) for eval captures. |
| `notebooks/test_bias_detection.py` | Interactive walkthrough of the same scenarios, designed to run in PyCharm or a notebook for visual inspection of baseline vs RAG output. |
| `scripts/record_response_run.py` | Connectivity check; runs the baseline slice from `scenarios_catalog.json` through `detect_bias_comparison`, attaches metrics + run card, writes `data/eval/runs/<timestamp>_<label>_rag_eval.json`. |
| `scripts/run_sample_comparison.py` | Runs two illustrative scenarios end-to-end; writes `data/eval/runs/sample_results.json` and **appends** JSONL rows (timestamp + `without_rag` / `with_rag` payloads) to `data/eval/runs/sample_results_comparison.md`. |
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

# Index-time chunk embeddings (build_vector_index.py). Queries always embed full scenario text.
RAG_EMBED_SENTENCE_WINDOWS=true   # false = one Titan call on full chunk text (legacy)
RAG_EMBED_SENTENCE_RADIUS=3       # sentences before / after each centre sentence in a window
RAG_EMBEDDING_MAX_WINDOWS=0       # 0 = no subsampling; set e.g. 24 to cap Bedrock calls per chunk

# Runtime overlap filter + neighbour expansion (rag/retriever.py)
RAG_OVERLAP_FILTER=false
RAG_OVERLAP_CHUNK_RADIUS=2
RAG_OVERLAP_POOL_MULTIPLIER=2
RAG_OVERLAP_MMR_POOL_MULTIPLIER=8
RAG_CONTEXT_EXPAND_NEIGHBORS=0
RAG_EXPAND_MAX_CHARS_PER_SIDE=450
```

`shared_components/settings.py` is the only place that reads these values. `BedrockProvider.__init__` validates that the bearer token is present before building a boto3 client and appends a connectivity log row on success.

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

Run unit tests (no Bedrock calls):

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
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

---

## 10. QA and Testing

### 10.1 Automated unit tests

| Item | Detail |
|------|--------|
| **Command** | `python -m unittest discover -s tests -p 'test_*.py' -v` (project root; no extra packages beyond `requirements.txt`). |
| **What is covered** | `tests/test_retriever_overlap.py` — overlap detection (`_chunk_overlap`), greedy backfill filtering (`_filter_overlap_greedy`), and neighbour excerpt assembly (`_neighbor_snippets`) on `KnowledgeRetriever` using `__new__` + synthetic `metadata` (no Bedrock, no FAISS files). |
| **Where results go** | **Stdout only** — unittest prints per-test OK/FAIL; exit code `0` means all tests passed. No XML/HTML report is generated unless you add a runner. |

### 10.2 Connectivity and Bedrock smoke

| Item | Detail |
|------|--------|
| **Command** | `python -c "from evaluation.connectivity import run_connectivity_check; print(run_connectivity_check())"` or the connectivity phase of `scripts/record_response_run.py` (omit `--skip-connectivity`). |
| **What is covered** | Bearer token present, Titan embedding dimension check, vector store load for `CORPUS_NAME`, single `search` probe. |
| **Where results go** | **`data/eval/runs/connectivity_log.txt`** — append-only TSV lines: UTC ISO timestamp, `OK` or `FAIL`, short message. Each successful **`BedrockProvider()`** also appends an `OK` row (`bedrock_runtime_client_initialized …`). |

### 10.3 Scenario / end-to-end runs (paid LLM)

| Item | Detail |
|------|--------|
| **Sample comparison** | `python scripts/run_sample_comparison.py` (optional `--skip-connectivity`). Writes **`data/eval/runs/sample_results.json`** and **appends** JSONL rows to **`data/eval/runs/sample_results_comparison.md`**. |
| **Frozen baseline capture** | `python scripts/record_response_run.py --label <tag>` — writes **`data/eval/runs/<timestamp>_<label>_rag_eval.json`** (connectivity, run_card, scenarios, aggregate metrics) and appends to **`data/eval/runs/sample_results_comparison.md`** unless `--no-append-sample-log`. |

### 10.4 Local artefact summary

All paths below are under **`data/eval/runs/`** (see `get_response_dir()` in `path_utils.py`; legacy files may exist under `archived/response_captures/`):

- **`connectivity_log.txt`** — longitudinal connectivity and client-init audit.
- **`sample_results_comparison.md`** — header plus one JSON object per appended line (`timestamp`, `without_rag`, `with_rag`, optional `scenario_id`, `label`).
- **`*_rag_eval.json`** — full structured captures for regression or diff tooling.

For a concise architecture-level view of the same QA flow, see **`docs/ARCHITECTURE.md` → QA and Testing**.

---

## 11. Document updates (2026-05-14)

- **Runtime retrieval:** `rag/retriever.py` documents and implements overlap-aware primary selection (`RAG_OVERLAP_*`) and optional same-book neighbour excerpts (`RAG_CONTEXT_EXPAND_*`, `neighbor_blocks` on `RetrievalResult`).
- **Connectivity audit:** `BedrockProvider` appends a row to `data/eval/runs/connectivity_log.txt` on each successful boto3 client construction; `evaluation/connectivity.run_connectivity_check` still appends a summary row per full smoke run.
- **Sample results log:** `data/eval/runs/sample_results_comparison.md` is the canonical append-only log (JSONL lines after a short header); `evaluation/sample_results_log.migrate_docs_sample_if_present` moves a legacy `docs/` copy once.
- **Tests:** `tests/test_retriever_overlap.py` — run via **§10.1** above.
