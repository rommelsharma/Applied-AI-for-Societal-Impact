# Architecture

## End-to-end pipeline (corpus-aware, implemented)

```text
PDF in data/corpora/<corpus>/raw/                  # corpus ∈ {public, private, ...}
  -> pdf_parser.py
  -> data/corpora/<corpus>/parsed_text/{book}.txt
  -> data/corpora/<corpus>/parsed_text/{book}.meta.json
  -> chunker.py                                    # profile: dossier | book | auto
  -> data/corpora/<corpus>/chunks/chunks.json
  -> concept_extractor.py
  -> data/corpora/<corpus>/chunks/chunks_with_concepts.json
  -> enrich_chunks.py                              # passage_type, decision_phase, summary, keywords, …
  -> data/corpora/<corpus>/knowledge/knowledge_base.json
  -> build_vector_index.py                         # Titan v2 + sentence-window pooling (see below)
  -> data/corpora/<corpus>/vector_store/
       embeddings.npy, knowledge.index, index_metadata.json, manifest.json
```

### Index-time embeddings (`build_vector_index.py`)

By default each **chunk row** still produces **one** FAISS vector (same row alignment as `index_metadata.json`). The vector is **not** a single embed of raw `chunk["text"]` unless disabled:

1. Chunk `text` is split into **sentences** (simple `. ? !` boundary rules in `build_vector_index.py`).
2. For **each sentence index** *i*, a window string is formed: up to **`RAG_EMBED_SENTENCE_RADIUS`** sentences before *i*, sentence *i*, and up to **`RAG_EMBED_SENTENCE_RADIUS`** after (clamped to the chunk). Default radius is **3** → up to **7** sentences per window.
3. Each window is embedded with **Amazon Titan Text Embeddings v2** (`normalize=true`).
4. Window vectors are **mean-pooled** and **L2-normalised** again so **inner product == cosine** for `IndexFlatIP`.

Optional **`RAG_EMBEDDING_MAX_WINDOWS`** (>0) **evenly subsamples** windows per chunk to cap Bedrock cost during large rebuilds. Set **`RAG_EMBED_SENTENCE_WINDOWS=false`** to restore the legacy **one embed = full chunk text** behaviour.

`vector_store/manifest.json` records `embedding_sentence_windows`, `embedding_sentence_radius`, and `embedding_max_windows_per_chunk` for reproducibility.

## Runtime RAG path

Implemented in `rag/retriever.py` (search, MMR, overlap filter, neighbour snippets) and `app/services/bias_detector.py` (`format_retrieved_context`).

```text
knowledge_base.json (via vector_store/index_metadata.json)
  -> Bedrock Titan v2: embed user scenario (full text, not sentence-windowed)
  -> FAISS IndexFlatIP (or numpy dot fallback)
  -> optional concept/domain filters, optional Haiku reranker, MMR (wider MMR pool when overlap filter on)
  -> optional overlap-aware top-k filter + backfill from ranked pool (RAG_OVERLAP_FILTER)
  -> optional bounded neighbor-chunk excerpts for prompt context (RAG_CONTEXT_EXPAND_NEIGHBORS)
  -> format_retrieved_context → prompt assembly
  -> Bedrock Claude (baseline + RAG JSON comparison)
```

### Retrieval quality refinements (implemented)

These stages **do not** change the vector index; they are toggled via `shared_components/settings.py` (env vars in `.env.example`).

**1. Overlap-aware top-k filtering** (`RAG_OVERLAP_FILTER`, `RAG_OVERLAP_CHUNK_RADIUS`, pool multipliers)

- **Definition:** Two chunks overlap when they share the same **`source`** and **`chunk_index`** differs by at most the configured radius.
- **Mechanism:** After rerank (if any) and **MMR** over an enlarged pool (`RAG_OVERLAP_MMR_POOL_MULTIPLIER`), a **greedy pass** keeps the MMR order but skips chunks that overlap any already-kept primary; **backfill** from deeper in the list fills `top_k`, then allows overlaps only if the pool is exhausted.
- **Why:** Complements MMR (embedding-space diversity) with **document-structure** diversity: fewer adjacent duplicates from the same book in one prompt.

**2. Bounded context expansion** (`RAG_CONTEXT_EXPAND_NEIGHBORS`, `RAG_EXPAND_MAX_CHARS_PER_SIDE`)

- **Mechanism:** For each final primary hit, optional same-**`source`** neighbours at `chunk_index ± N` supply short excerpts (`RetrievalResult.neighbor_blocks`); `format_retrieved_context` labels them as **continuation** text so citations stay on the primary chunk ID.
- **Why:** Improves local coherence at bad chunk boundaries without extra embedding calls.

**Comparing before and after**

Use the same corpus, index (`manifest.json`), Bedrock models, and frozen scenarios; toggle only the env flags. Capture runs via `scripts/record_response_run.py` into `response/` with distinct labels. `evaluation/run_card.py` records overlap and expansion settings. Compare distinct sources, adjacency among primaries, and qualitative JSON — alongside append-only `response/sample_results_comparison.md` (JSONL rows) and full `response/*_rag_eval.json` captures.

## Evaluation and response capture (implemented)

```text
evaluation/baseline_scenarios.json
  -> scripts/record_response_run.py (connectivity + detect_bias_comparison + metrics + run_card)
  -> response/<timestamp>_<label>_rag_eval.json
```

Supporting modules: `evaluation/connectivity.py`, `evaluation/metrics.py`, `evaluation/run_card.py`, full suite in `evaluation/run_evaluation.py`.

## QA and Testing

**Automated unit tests (no Bedrock, no vector store required for overlap helpers)**

- **Command:** from the project root, `python -m unittest discover -s tests -p 'test_*.py' -v`
- **Scope:** `tests/test_retriever_overlap.py` exercises `_chunk_overlap`, `_filter_overlap_greedy`, and `_neighbor_snippets` on `KnowledgeRetriever` (instantiated via `__new__` with mock metadata — no FAISS or credentials).
- **Results:** pass/fail and tracebacks are **only on the console** (stdlib unittest does not write a JUnit file unless you wrap it). CI or local runs should treat exit code `0` as success.

**Integration / smoke (Bedrock + corpus index)**

- **Connectivity:** `python -c "from evaluation.connectivity import run_connectivity_check; print(run_connectivity_check())"` or rely on the first step inside `scripts/record_response_run.py` when not using `--skip-connectivity`. Validates credentials, Titan embedding round-trip, vector store load, and a short `KnowledgeRetriever.search`.
- **Where logged:** each successful `BedrockProvider()` construction appends one tab-separated row to `response/connectivity_log.txt`; each full `run_connectivity_check()` appends a final summary row (OK/FAIL + detail). Inspect that file for an audit trail over time.

**Scenario-level QA (LLM calls, costs real tokens)**

- **`scripts/run_sample_comparison.py`** — two fixed scenarios; writes `evaluation/sample_results.json` and **appends** one JSON line per scenario to `response/sample_results_comparison.md` (timestamp + `without_rag` / `with_rag` payloads after the file header).
- **`scripts/record_response_run.py`** — runs `evaluation/baseline_scenarios.json` (three scenarios by default), metrics, run card; writes `response/<timestamp>_<label>_rag_eval.json` and can append the same JSONL log (unless `--no-append-sample-log`).

**Where results are stored (local `response/` directory)**

| Artefact | Purpose |
|----------|---------|
| `response/connectivity_log.txt` | Append-only connectivity / Bedrock client events (TSV: UTC timestamp, OK/FAIL, message). |
| `response/sample_results_comparison.md` | Append-only log: markdown header, then **one JSON object per line** per scenario batch (includes `without_rag` and `with_rag`). |
| `response/<timestamp>_<label>_rag_eval.json` | Full eval capture from `record_response_run.py` (connectivity payload, run_card, per-scenario results and metrics). |

Re-run the same commands after changing `RAG_OVERLAP_*` or `RAG_CONTEXT_EXPAND_*` to compare behaviour; keep `manifest.json` and scenario files fixed when doing A/B retrieval checks.

## Repository layout (high level)

```text
bias-aware-decision-making-rag-copilot/
├── data/corpora/<corpus>/{raw,parsed_text,chunks,knowledge,vector_store}
├── data/metadata/                  # bias-taxonomy, retrieval-concepts, …
├── data_pipeline/
├── shared_components/
├── rag/
├── prompts/
├── evaluation/
├── response/                       # eval JSON, connectivity_log.txt, sample_results_comparison.md (append-only)
├── scripts/
└── README.md
```

## Important design choices

### Metadata sidecar files

Each parsed book should produce both:

- a text file with extracted content
- a JSON sidecar with metadata such as author, title, publication year, and parser notes

### Ingestion cleanup

Chunking should trim front matter and back matter so retrieval is grounded in actual book content rather than copyright pages, tables of contents, notes, or author bios.

### Traceability in chunks

Every chunk should carry:

- unique id
- source file or book slug
- author
- title
- publication year
- chunk index
- word count
- extracted concepts
- cleaning notes
- confidence/importance fields when available

### Taxonomy-driven enrichment

Concept extraction should be driven by external metadata files rather than a small hardcoded keyword map:

- `data/metadata/bias-taxonomy.json`
- `data/metadata/retrieval-concepts.json`

This keeps retrieval concepts and prompt taxonomy aligned.

### Centralized path management

Use shared utilities for:

- project root resolution
- raw data directory
- processed data directory
- ensuring directories exist

### Direct execution support

Scripts should work when run from:

- PyCharm
- terminal
- module invocation like `python -m data_pipeline.build_knowledge_base`

## RAG roadmap (remaining enhancements)

1. Hybrid lexical + dense retrieval (BM25 + vector).
2. Query decomposition for multi-part scenarios.
3. Richer eval: gold chunk labels, retrieval-precision metrics, CI regression on frozen baselines.
4. Optional scaling: `IndexIVFFlat` / HNSW when chunk counts grow substantially.
5. Tune overlap radius and MMR pool multipliers against frozen scenarios; document deltas in `response/sample_results_comparison.md` captures.

---

## Document updates (2026-05-14)

Runtime overlap filtering and bounded neighbour expansion are **implemented** (env-gated) in `rag/retriever.py` and reflected in `format_retrieved_context` / retrieval payloads. `response/connectivity_log.txt` records each Bedrock client construction and each `run_connectivity_check` outcome. `response/sample_results_comparison.md` is the append-only scenario log (JSONL). **QA and Testing** (this document, previous section) and **`docs/code-flow.md` §10–11** describe how tests were run and where outputs land.
