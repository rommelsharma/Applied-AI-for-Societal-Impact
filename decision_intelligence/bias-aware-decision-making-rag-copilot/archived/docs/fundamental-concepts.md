# Fundamental Concepts

This document explains the *why* behind the Bias-Aware Decision Making RAG Copilot — the logical flow, the core ideas in retrieval-augmented generation, and the rationale for the specific technologies chosen.

It is intended as a primer for engineers, reviewers, and HR / Legal / business stakeholders who want to understand not just what the system does but *why* it is built this way.

---

## 1. The Logical Flow at a Glance

A user submits a complex, real-world scenario. The system responds with a structured analysis grounded in curated decision-science literature. Two parallel reasoning paths are executed and compared:

```text
                ┌──────────────────────────────────────────┐
                │             User scenario text           │
                └──────────────────────────────────────────┘
                                     │
                  ┌──────────────────┴───────────────────┐
                  ▼                                      ▼
       BASELINE PATH                          RAG-ENHANCED PATH
  (taxonomy + LLM only)                  (taxonomy + retrieval + LLM)
                  │                                      │
                  │                       ┌──────────────┴───────────────┐
                  │                       │  Concept extraction on text   │
                  │                       │  Embed scenario (Titan v2)   │
                  │                       │  FAISS top-k similarity       │
                  │                       │  Concept / domain filters     │
                  │                       │  MMR diversification          │
                  │                       │  overlap-aware filter (opt.)    │
                  │                       │  neighbour expansion (opt.)     │
                  │                       │  Format context blocks        │
                  │                       └──────────────┬───────────────┘
                  │                                      │
                  ▼                                      ▼
       Claude Sonnet 4.5                       Claude Sonnet 4.5
       (no retrieved context)                  (with retrieved context)
                  │                                      │
                  ▼                                      ▼
          Strict JSON answer                     Strict JSON answer
                  │                                      │
                  └──────────────────┬───────────────────┘
                                     ▼
                  Comparison payload {without_rag, with_rag, retrieval}
```

The comparison structure is deliberate: it makes the lift from RAG measurable and demonstrable.

---

## 2. Retrieval-Augmented Generation (RAG)

### What it is

RAG is a pattern in which an LLM's prompt is enriched at runtime with passages retrieved from an external knowledge base. The LLM does not rely solely on its training data — it grounds its reasoning in passages the operator controls.

### Why this project needs it

The copilot's value proposition depends on three properties that a vanilla LLM cannot reliably provide:

1. **Source attribution.** Every claim should be traceable to a named book, author, and chunk.
2. **Curatorial control.** The knowledge base must be a deliberate selection of decision-science literature, not whatever the model happened to memorise.
3. **Update without retraining.** Adding *Behave* or *The Black Swan* tomorrow should be a corpus operation, not a model operation.

RAG gives all three.

### What gets retrieved

Not raw books — a structured, traceable derivative:

- ~300 chunks for the **public** corpus (synthesised dossiers + the *Noise* book at ~500 words each).
- ~1,160 chunks for the **private** corpus (six full books at ~500 words each, local-only).
- Each chunk carries source metadata, concept tags, decision domains, importance, chapter title, passage type, decision phase, and a short summary.
- The chunk text plus its metadata is what the LLM sees in the system prompt.

### 2.1 Overlap-aware top-k and bounded context expansion (implemented)

Two retrieval refinements run after semantic search, filters, optional reranking, and **MMR** in `rag/retriever.py`, with neighbour text rendered in `format_retrieved_context`. They are **not a substitute** for MMR; they address different failure modes. Both are **env-gated** (see `.env.example`).

**Overlap-aware top-k.** MMR penalises chunks whose **embedding vectors** are too similar to chunks already chosen. It does not explicitly encode **document structure**: the same book and **adjacent `chunk_index`** values can still yield several hits that paraphrase the same passage. When `RAG_OVERLAP_FILTER=true`, a **greedy filter** treats two chunks as overlapping when they share the same `source` and `chunk_index` lies within **`RAG_OVERLAP_CHUNK_RADIUS`**; the retriever skips overlapping primaries and **backfills** from deeper in the MMR-ordered pool (widened by `RAG_OVERLAP_MMR_POOL_MULTIPLIER` and a larger FAISS pool via `RAG_OVERLAP_POOL_MULTIPLIER`) until `top_k` is filled.

**Bounded expansion.** Index-time **sentence-window pooling** already enriches each vector. At **runtime**, optional **`RAG_CONTEXT_EXPAND_NEIGHBORS`** loads same-`source` rows at `chunk_index ± N` from `index_metadata.json`, capped by **`RAG_EXPAND_MAX_CHARS_PER_SIDE`**, and appends excerpts under **continuation** labels so citations stay on the **primary** chunk ID.

**Evaluation discipline.** Compare runs with the same frozen index and scenarios, toggling only these env flags; store timestamped JSON under `data/eval/runs/` and append JSONL rows to `data/eval/runs/sample_results_comparison.md`.

---

## 3. Embeddings

### What an embedding is

An embedding is a fixed-length numeric vector that represents the *meaning* of a text snippet. Two snippets with similar meaning produce vectors that point in similar directions in high-dimensional space, even when they share no exact words. "A manager favoured a candidate from a similar background" and "in-group preference in hiring" embed close together; the words barely overlap, the meaning does.

### Why this matters here

User scenarios are paraphrases. They never quote the books. Keyword search would miss almost everything relevant. Semantic search via embeddings is the only retrieval strategy that survives paraphrase.

### How they are produced in this project

**Query side (runtime):** the user’s full scenario string is embedded once with **Amazon Titan Text Embeddings v2** (`amazon.titan-embed-text-v2:0`) at 1024 dimensions with `normalize=true`, via `BedrockProvider.embed_text()`.

**Corpus side (index build):** `data_pipeline/build_vector_index.py` embeds each **knowledge-base chunk** as follows (unless `RAG_EMBED_SENTENCE_WINDOWS=false`):

1. Split chunk `text` into sentences (simple punctuation-based splitter).
2. For each sentence index *i*, form a **window** of up to **`RAG_EMBED_SENTENCE_RADIUS`** sentences before *i*, sentence *i*, and up to **`RAG_EMBED_SENTENCE_RADIUS`** after (default radius **3** → up to **seven** sentences, fewer at edges).
3. Embed each window with Titan (`normalize=true`). Optionally subsample windows when `RAG_EMBEDDING_MAX_WINDOWS` > 0 to cap Bedrock cost.
4. **Mean-pool** the window vectors and **L2-normalise** the result so each chunk still occupies **one row** in `embeddings.npy` / FAISS—inner product search remains **cosine-equivalent** on unit vectors.

Normalised vectors mean inner product equals cosine similarity, which is what FAISS `IndexFlatIP` uses internally.

### Why not sentence-transformers locally?

- Titan v2 is hosted, managed, and consistent with the inference plane (Bedrock), which keeps the operational footprint small.
- It removes the need to host or fine-tune an embedding model just for a portfolio system.
- Switching providers later is a one-line change because all embedding access goes through `BedrockProvider.embed_text()`.

---

## 4. Why FAISS

[FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) is the de-facto standard library for similarity search over dense vectors.

### What FAISS does

Given a set of N vectors of dimension D, FAISS pre-organises them into a data structure that answers queries of the form *"find the k vectors closest to this query vector"* in milliseconds, even when N is in the millions.

### How it works (the version used here)

This project uses `IndexFlatIP`:

- **Flat** — every vector is stored verbatim and the search is exhaustive.
- **IP** — the comparison metric is **inner product**. Combined with normalised embeddings, this is mathematically equivalent to cosine similarity.

For a few hundred chunks (the public dossier index is on the order of **300** rows), exhaustive search is not just acceptable — it is preferable: zero approximation error, deterministic ordering, no training step. When the corpus grows past tens of thousands of chunks, the index can be swapped for `IndexIVFFlat` or `IndexHNSWFlat` without changing any other code, because the retriever only depends on the `search(query_vector, k)` interface.

### Why FAISS over alternatives

| Option | Why not (for this project) |
|--------|----------------------------|
| Pinecone, Weaviate, Qdrant Cloud | Network round-trips, billing setup, and a managed service are overkill for a portfolio-scale corpus. |
| pgvector | Excellent at scale, but adds a Postgres dependency for a few hundred vectors. |
| Pure numpy (dot product) | Works (and is the fallback), but has no path to scale or to ANN indexes. |
| FAISS `IndexFlatIP` | Zero approximation, microsecond latency at this size, drop-in upgrade path to ANN indexes later. |

A numpy-only fallback path is preserved in `rag/retriever.py` so contributors who do not have FAISS installed locally can still run the system.

---

## 5. Why Amazon Bedrock

### What Bedrock is

Bedrock is AWS's managed inference service that exposes foundation models (Anthropic Claude, Amazon Titan, Meta Llama, Mistral, Cohere, AI21) through a single, consistent API. It handles model hosting, scaling, observability, IAM-based authorisation, and regional availability.

### Why Bedrock for this project

1. **One control plane for chat and embeddings.** Both Claude (chat) and Titan (embeddings) are reachable through the same boto3 client, the same IAM identity, and the same VPC pattern. This keeps `BedrockProvider` tiny and the operational story simple.
2. **Enterprise posture.** The portfolio is intentionally pitched as enterprise-grade. Bedrock fits the way large organisations actually deploy generative AI — VPC endpoints, KMS, CloudTrail, no model-vendor sprawl, no separate billing relationships per provider.
3. **Model swap without code changes.** If Claude Sonnet 4.5 is replaced by Claude 5 or a Titan Premier release, only the model id in `.env` changes.
4. **No data leakage to model providers.** Bedrock does not use customer prompts to train foundation models, which matters for any decision-support system that may be exposed to sensitive scenarios.
5. **Aligned with the AWS-native stack** the project assumes for hosting and MLOps (S3 for the vector store, CloudWatch for observability, IAM for auth).

### What is not used

OpenAI, Anthropic API directly, Google Vertex, and Azure OpenAI are all reasonable alternatives. They were not chosen because the project's positioning is enterprise-AWS and because using two providers (one for chat, one for embeddings) would double the credential and auditing surface for no real benefit.

---

## 6. Why Claude Sonnet 4.5 for Reasoning

The bias-detection task requires:

- **Long, structured JSON output** without truncation.
- **Faithful adherence to a strict schema** and a closed taxonomy (the model must not invent biases).
- **Nuanced reasoning** about human and systemic factors in a scenario.
- **Consistency across two calls** (baseline and RAG) so the comparison is meaningful.

Claude Sonnet 4.5 is well-suited on all four axes, and Bedrock exposes it via the Converse API which standardises system prompts, message turns, and inference parameters. Temperature is set to `0.1` to maximise determinism for evaluation runs.

---

## 7. Worked Example — One Book Through the Whole Pipeline

The previous sections explain *why* each tool is used. This section shows *how* they fit together by following one real book end-to-end through the system: Olivier Sibony's *You're About to Make a Terrible Mistake* (2019). The book sits in the **private** corpus because the project distinguishes synthesised public dossiers from full local-only book PDFs (see [`PRIVATE_CORPUS_GUIDE.md`](PRIVATE_CORPUS_GUIDE.md)).

All numbers below are real values produced by the live ingestion run.

### Step 0 — The book lands on disk

```text
data/corpora/private/raw/Youre_About_to_Make_a_Terrible_Mistake_-_Olivier_Sibony.pdf   3.17 MB
```

The path itself encodes that the file is in the **private** corpus, so every downstream tool reads and writes its derivatives under `data/corpora/private/`. Nothing in this directory is committed to git.

### Step 1 — PDF parsing (`data_pipeline/pdf_parser.py`)

PyMuPDF extracts the body text and infers metadata from the first few pages:

```text
pages          : 279
characters     : 512,426
words          : 82,086
parser_notes   : ["author_not_found_in_first_pages",
                  "publication_year_not_found_in_first_pages"]
```

`parser_notes` is recorded on the metadata sidecar so reviewers can immediately see what the heuristics could not infer. The text is also cleaned for soft-hyphenation breaks (`commit-\nment` → `commitment`) so chunk boundaries do not split words.

### Step 2 — Chunking (`data_pipeline/chunker.py`)

The chunker auto-detects this PDF as a *book* (not a dossier) — at 82k words it crosses the 30k auto-threshold — and applies the **book profile** (500-word windows with 80-word overlap). Front- and back-matter trimming uses line-anchored regex headings so inline citations like "see notes" or "index" inside the body cannot misfire.

```text
chunking_profile           : book
chunks created             : 177
trim retained              : 85.3% of original text
chapter_title              : populated for chunks that fall inside a detected
                              chapter heading
```

Each chunk inherits a stable id (`Youre_About_to_Make_a_Terrible_Mistake_-_Olivier_Sibony_chunk_137`), the source filename, parser/cleaning notes, and chapter title.

### Step 3 — Concept tagging (`data_pipeline/concept_extractor.py`)

The taxonomy (`bias-taxonomy.json`) and the support-concepts catalog (`retrieval-concepts.json`) are loaded and aliases are matched against each chunk's text. The eight most frequent tags surfaced from this single book:

| Concept | Hits in Sibony | What it captures |
|---|---|---|
| `estimate_talk_estimate` | 73 | Independent-then-aggregate forecasting |
| `judgment` | 60 | General judgment-quality discussions |
| `structured_analytic_techniques` | 47 | Devil's-advocate, key-assumptions check |
| `groupthink` | 36 | Conformity-driven collective decisions |
| `confirmation_bias` | 28 | Selectively gathering corroboration |
| `anchoring_bias` | 16 | Initial-number influence |
| `uncertainty_management` | 13 | Reasoning under deep uncertainty |
| `loss_aversion` | 11 | Asymmetric weighting of gains vs losses |

The concept tags are what give every chunk a *symbolic* handle in addition to its dense vector — used later for filtered retrieval and for keeping the LLM constrained to a closed bias vocabulary.

### Step 4 — Enrichment (`data_pipeline/enrich_chunks.py`)

Each chunk is then classified for `passage_type` and `decision_phase` by the lightweight rule-based classifier (LLM-upgradable via `ENRICHMENT_USE_LLM=true`). The Sibony distribution — heavily prescriptive, decide-phase dominant — matches what you would expect from a practitioner-oriented book on better business decisions:

```text
passage_type : prescription 87, study 56, framework 18, anecdote 7, case_study 4, definition 2, unknown 3
decision_phase: decide 116, diagnose 25, design 13, review 6, unknown 17
```

The enriched record for a single Sibony chunk looks like this (truncated for clarity):

```json
{
  "id": "Youre_About_to_Make_a_Terrible_Mistake_-_Olivier_Sibony_chunk_137",
  "source": "Youre_About_to_Make_a_Terrible_Mistake_-_Olivier_Sibony",
  "author": "unknown",
  "chapter_title": "chapter 16 suggests twelve techniques to foster productive dynamics at",
  "chunking_profile": "book",
  "passage_type": "study",
  "decision_phase": "decide",
  "concepts": ["confirmation_bias", "groupthink", "hindsight_bias",
               "premortem", "uncertainty_management"],
  "decision_domains": ["governance", "leadership", "risk", "strategy"],
  "summary": "...",
  "text": "...",
  "parser_notes": ["author_not_found_in_first_pages",
                   "publication_year_not_found_in_first_pages"]
}
```

This is the row that the embedder reads next.

### Step 5 — Bedrock embeddings (`amazon.titan-embed-text-v2:0`)

At **index build** time, `build_vector_index.py` does **not** simply embed the entire chunk string once (unless `RAG_EMBED_SENTENCE_WINDOWS=false`). The default path is:

```text
1. Split chunk text into sentences.
2. For each sentence index i, build a window: up to R sentences before, sentence i, up to R after
   (R = RAG_EMBED_SENTENCE_RADIUS, default 3 → up to 7 sentences per window).
3. Embed each window with Titan v2, normalize=true.
4. Mean-pool window vectors, L2-normalise → one 1024-dim row per chunk in embeddings.npy.
```

So the **input** to the pooling step is many short, locally coherent strings; the **stored** vector is still one row per chunk, aligned with `index_metadata.json`.

```text
per-window input : up to ~7 sentences (~tens to low-hundreds of words)
per-chunk output   : numpy array of shape (1024,), dtype float32, unit L2 norm after pooling
cost               : ~one Titan call per sentence-window per chunk (use RAG_EMBEDDING_MAX_WINDOWS to cap)
```

The unit-norm property is what lets us treat **inner product** as **cosine similarity** in the next step — they are mathematically equivalent for normalised vectors.

For the full private corpus the embedder produces a stacked matrix:

```text
data/corpora/private/vector_store/embeddings.npy
shape : (1161, 1024)
dtype : float32
size  : 4.54 MB
```

The Sibony book contributes 177 of those 1,161 rows.

### Step 6 — FAISS index build

The same matrix is wrapped into a FAISS `IndexFlatIP`:

```python
import faiss, numpy as np
embeddings = np.load("embeddings.npy")            # (1161, 1024)
index = faiss.IndexFlatIP(1024)
index.add(embeddings)                             # exhaustive flat index, no training
faiss.write_index(index, "knowledge.index")
```

What FAISS gives this project, very concretely:

| Property | Value for this corpus |
|---|---|
| Index type | `IndexFlatIP` (exact, no approximation) |
| Vectors stored | 1,161 |
| Dimension | 1,024 |
| On-disk size | 4.54 MB (`knowledge.index`) |
| Aligned metadata | 5.31 MB (`index_metadata.json`, same row order) |
| Query latency | sub-millisecond at this size |
| Cost at runtime | $0 (local file, no service call) |

Because the index, the embeddings matrix, and `index_metadata.json` all share the same row order, a hit at row `882` in FAISS gives us back the Sibony chunk with all of its metadata in one lookup.

### Step 7 — Public vs private vector indexes coexisting

Both corpora produce the same five artefacts (`embeddings.npy`, `knowledge.index`, `index_metadata.json`, `manifest.json`, `chunks/...`), but under different parents:

```text
data/corpora/public/vector_store/   →  301 vectors  (committed dossiers + Noise)
data/corpora/private/vector_store/  →  1,161 vectors (6 full books, local-only)
```

The retriever picks one or the other via `corpus="public" | "private"` (or the `CORPUS_NAME` env var). `scripts/compare_versions.py` runs the same scenario suite against both indexes and writes a markdown summary of the lift, which is how the project keeps its public-vs-private claims honest.

### Step 8 — A live query reaches the Sibony chunks

Consider the question:

> "Before approving a strategic decision, how should the team systematically search for what could go wrong?"

The runtime path is:

1. **Embed the query** — Titan v2 on the **full** scenario text (same API as index time, but no sentence-window pooling on the query).
2. **FAISS top-k search** — inner-product search returns the 24 candidate rows whose vectors point closest to the query vector. The raw top-5 (cosine scores) for this query against the private index:

   ```text
   row=882   score=0.4498   Sibony       chapter 16 (twelve techniques to foster productive dynamics)
   row=865   score=0.4495   Sibony       chapter 16
   row=1113  score=0.4455   Noise        CHAPTER 25 (The Mediating Assessments Protocol)
   row=747   score=0.4367   Sibony       chapter 4
   row=832   score=0.4214   Sibony       chapter 6
   ```

3. **Concept / domain filtering** — the user's scenario is run through the same concept extractor, and chunks that share concepts with the scenario are preferred (with a fallback to unfiltered semantic results so the retriever never returns empty).
4. **MMR diversification** — Maximal Marginal Relevance with `λ=0.7` re-orders candidates so a single book or chapter cannot monopolise the result set. The Noise book's *Mediating Assessments Protocol* chunk gets pulled into the final top-5 even though three Sibony chunks scored higher, because MMR penalises near-duplicates.
5. **Optional Claude-Haiku reranker** — disabled by default; a place to plug in an LLM-as-judge stage once the corpus exceeds ~3,000 chunks.
6. **Format as system-prompt context** — each retrieved chunk is rendered as a numbered block with source, author, chapter, passage type, decision phase, concepts, summary, and a 900-character excerpt. That bundle is concatenated into the system prompt and sent to Claude Sonnet 4.5.

**Optional (env-gated):** **overlap-aware filtering** after MMR suppresses multiple primaries from the same `source` within a small `chunk_index` window, with backfill from the ranked pool; **bounded neighbour expansion** adds same-book excerpts for prompt coherence only (primary chunk IDs remain the citation anchors). See [`ARCHITECTURE.md`](ARCHITECTURE.md) and §2.1 above.

### Why this end-to-end matters

| Tool | What it actually buys this project |
|---|---|
| **Bedrock Titan v2 embeddings** | Robust paraphrase-tolerant retrieval without training or hosting an embedding model. Same boto3 client as the chat model. Unit-norm vectors fall straight into cosine search. |
| **FAISS `IndexFlatIP`** | Exact, deterministic, sub-millisecond top-k over the whole corpus. Zero approximation error at this scale. Drop-in upgrade path to ANN indexes (`IndexIVFFlat`, `IndexHNSWFlat`) once the corpus crosses tens of thousands of chunks. |
| **Public + private vector indexes coexisting** | The public index is small, attributable, and safe to ship in a demo. The private index is the *measurement* artefact: it shows the lift the system actually reaches with full-text books, without any of those books being committed to a public repo. |
| **Bedrock Claude Sonnet 4.5** | Long structured JSON output, faithful schema adherence, deterministic enough at temperature 0.1 to make the baseline-vs-RAG comparison meaningful. |

The Sibony book is a useful illustration because it crosses every boundary in the system: it is a full book (so it triggers the *book* chunking profile), it sits in the *private* corpus (so its derivatives never enter git), and it produces 177 chunks that compete with content from *Noise*, *Thinking, Fast and Slow*, and three other books for retrieval slots — exactly the conditions under which MMR diversification and the public-vs-private corpus split start to pay off.

---

## 8. Why a Taxonomy-Driven Concept Layer

A pure-vector RAG system is fragile because:

- Two unrelated chunks can embed close together and crowd out the genuinely relevant one.
- The LLM has no structured handle on *what kind of bias* a chunk is about.

This project adds a **taxonomy** (`bias-taxonomy.json`) and a **support-concepts catalog** (`retrieval-concepts.json`), both data-driven JSON files. They are used in three places:

1. **Indexing time** — every chunk is tagged with the concepts whose canonical name or aliases appear in its text (`concept_extractor.py`).
2. **Retrieval time** — the user's scenario is run through the same concept extractor; the retriever can then prefer chunks that overlap on concept tags or decision domains.
3. **Generation time** — the full taxonomy is injected into the LLM's system prompt so the model is constrained to a closed set of named biases.

This hybrid (semantic search + symbolic taxonomy) is a key part of the explainability story.

---

## 9. Why Strict JSON Output

The system prompt locks the model into a JSON contract:

```json
{
  "situation_summary": "...",
  "biases_identified": [{"bias_name": "...", "bias_layer": "...", "confidence": "...", "explanation": "...", "evidence": "...", "risk": "..."}],
  "noise_considerations": {"present": true, "explanation": "..."},
  "overall_risk_assessment": "...",
  "recommended_actions": [...],
  "alternative_perspectives": [...]
}
```

Why:

- It is machine-parseable, so any UI, evaluator, or downstream service can render it deterministically.
- It eliminates rambling prose that obscures whether the model actually grounded its answer.
- It enables programmatic comparison between baseline and RAG outputs (e.g. counting biases identified, checking schema fidelity).

`bias_detector.extract_json_payload()` includes a tolerant parse path (strips fenced code blocks, extracts the outermost JSON object) and a one-shot stricter retry if the first response is malformed.

---

## 10. Why Profile-Driven Chunking (Dossier vs Book)

- **Dossier profile (350 / 60).** Synthesised dossiers are concept-dense and short — 350 words is roughly the length of a single coherent argument, and a 60-word overlap keeps key sentences from being split between adjacent chunks.
- **Book profile (500 / 80).** Full-length books are anecdote-rich rather than concept-dense, so they need more context per chunk to keep a worked example or named study together with its surrounding framing. The larger 80-word overlap absorbs longer sentences.
- **Auto profile.** Documents above ~30,000 words automatically use the *book* profile, otherwise *dossier*. This means a single corpus can hold both syntheses and full books and the right profile is picked per document.

`chunker.PROFILES` and the auto-threshold are tunable; the chosen profile is stamped onto every chunk (`chunking_profile`) so retrieval lift can later be analysed by profile.

---

## 11. Why Front- and Back-Matter Trimming

Books carry copyright pages, dedications, tables of contents, acknowledgments, notes, indexes, and "also by this author" sections. None of those are useful for retrieval, and they actively pollute the embedding space — copyright boilerplate looks lexically similar across books and steals retrieval slots from real content.

`chunker.trim_document_text()` uses **line-anchored regex heading detection** rather than naive substring search, because words like `notes`, `index`, and `references` appear as inline citations throughout the body of academic-style books and would otherwise misfire. Front-matter markers are searched in the first 60% of the document; back-matter markers only in the last 25%. A safety net rejects any trim that would discard more than 70% of the document and records `trim_skipped_too_aggressive` in `cleaning_notes` so the case is visible to reviewers.

---

## 12. Why a Baseline-vs-RAG Comparison Mode

The single most important question an HR, Legal, or business reviewer will ask is: *"Does the retrieval actually help?"*

Running both prompts on every scenario answers that question with evidence rather than belief. Over time, the comparison can be scored on:

- Number and quality of biases identified.
- Whether the model cites a retrieved source.
- Specificity of evidence (does the model quote the scenario or wave its hands?).
- Schema fidelity and absence of hallucinated bias names.

This is also what turns the project from "demo" into "evaluable system."

---

## 13. Why Centralised Configuration and Path Resolution

Two small but important design choices:

- `shared_components/settings.py` is the **only** module that reads environment variables. Every other file imports `BEDROCK_SETTINGS`. Rotating a region or switching a model never requires touching application code.
- `shared_components/utilities/path_utils.py` is the **only** module that resolves paths. No file uses `Path(__file__).parents[3]` or hard-coded strings. Moving a directory becomes a one-line change.

These conventions are unglamorous but they are what keep the codebase explainable as it grows.

---

## 14. Why Traceability Is a First-Class Citizen

The system's domain is decision support for high-stakes human contexts (hiring, compliance, performance review). Any answer the model gives must be auditable. The repository enforces this via:

- Metadata sidecars on every parsed PDF.
- `cleaning_notes` and `parser_notes` recorded on every chunk.
- Concept and decision-domain tags carried from chunk → embedding metadata → retrieval result → final response payload.
- A retained `retrieval` array in every comparison output so the calling layer can show *which* sources shaped the answer.

This is the same pattern that enterprise AI governance reviews look for, and it is one of the project's deliberate enterprise-style signals.

---

## 15. What This System Is Not

To set expectations honestly:

- It is **not** legal, medical, HR, or financial advice. Outputs are research-informed perspectives, not rulings.
- It is **not** a replacement for professional judgment or organisational policy.
- It is **not** a finished product — it is a portfolio-grade, evaluable foundation that is intentionally explainable end-to-end.

Everything in the design — the taxonomy, the strict JSON, the comparison mode, the source attribution — exists so the system can be honest about what it knows and where that knowledge came from.

---

## QA and Testing (how we validate behaviour)

This section complements the **worked Sibony example** (runtime retrieval in practice): it states *how* the repository is tested and *where* evidence is kept.

**1. Fast, local unit tests (no cloud spend)**  
Run from the project root:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

These tests target the **overlap filter** and **neighbour expansion** helpers in `rag/retriever.py` (`tests/test_retriever_overlap.py`). They use synthetic chunk metadata only. **Results:** printed on the terminal; **exit code `0`** means success. Nothing is written under `data/eval/runs/` for this step.

**2. Connectivity smoke (Bedrock + index on disk)**  
`evaluation.connectivity.run_connectivity_check()` embeds a probe string, loads the active corpus vector store, and runs one retrieval. It is invoked directly or as the first step of `scripts/record_response_run.py`. **Results:** a structured dict on stdout when run inline; **append-only** lines in **`data/eval/runs/connectivity_log.txt`** (timestamp, OK/FAIL, short detail). Each new **`BedrockProvider()`** also appends one line when the boto3 client is created successfully.

**3. Scenario-level runs (baseline vs RAG, uses Claude)**  
- `scripts/run_sample_comparison.py` — two illustrative scenarios; appends one **JSON line per scenario** to **`data/eval/runs/sample_results_comparison.md`** (after the header) and saves raw payloads to **`data/eval/runs/sample_results.json`**.  
- `scripts/record_response_run.py` — three frozen scenarios by default (from `data/eval/gold/scenarios_catalog.json`); writes **`data/eval/runs/<timestamp>_<label>_rag_eval.json`** and can append the same markdown log.

**4. Where to look after a QA pass**

| Location | What it contains |
|----------|------------------|
| `data/eval/runs/connectivity_log.txt` | TSV audit trail for connectivity checks and Bedrock client construction. |
| `data/eval/runs/sample_results_comparison.md` | Append-only JSONL (each line: `timestamp`, `without_rag`, `with_rag`, …). |
| `data/eval/runs/*_rag_eval.json` | Full capture: connectivity report, run_card, per-scenario metrics and payloads. |

For file-level navigation of scripts and modules, see **`docs/code-flow.md` §10–11**; for a pipeline-level summary, see **`docs/ARCHITECTURE.md` → QA and Testing**.

---

## Document updates (2026-05-14)

Section **2.1** and the **§1 diagram** now describe **implemented** overlap-aware filtering and neighbour expansion (env-gated). **Step 8** in the Sibony walkthrough notes the same optional stages. **QA and Testing** (section above) documents unittest, connectivity, scenario scripts, and `data/eval/runs/` artefacts; cross-links to `docs/code-flow.md` and `docs/ARCHITECTURE.md`.
