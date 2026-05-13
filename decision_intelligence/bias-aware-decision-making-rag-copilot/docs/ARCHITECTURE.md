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

## Runtime RAG path (implemented)

```text
knowledge_base.json (via vector_store/index_metadata.json)
  -> Bedrock Titan v2: embed user scenario (full text, not sentence-windowed)
  -> FAISS IndexFlatIP (or numpy dot fallback)
  -> optional concept/domain filters, optional Haiku reranker, MMR
  -> format_retrieved_context → prompt assembly
  -> Bedrock Claude (baseline + RAG JSON comparison)
```

## Evaluation and response capture (implemented)

```text
evaluation/baseline_scenarios.json
  -> scripts/record_response_run.py (connectivity + detect_bias_comparison + metrics + run_card)
  -> response/<timestamp>_<label>_rag_eval.json
```

Supporting modules: `evaluation/connectivity.py`, `evaluation/metrics.py`, `evaluation/run_card.py`, full suite in `evaluation/run_evaluation.py`.

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
├── response/                       # timestamped eval captures (.gitkeep + local JSON)
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
