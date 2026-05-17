# Bias-Aware Decision Making RAG Copilot

**A research-grounded AI assistant that helps people, teams, and HR functions make fairer, more consistent decisions — by surfacing the cognitive and systemic biases hiding inside everyday business situations.**

Built on Anthropic Claude via Amazon Bedrock, with a curated decision-science knowledge base, a closed bias taxonomy, full source attribution, and strict structured output that integrates cleanly into existing review and case-management workflows.

> Decision support only. Not legal, medical, HR, or financial advice. Always pair with human judgment and domain expertise.

---

## Why This Exists

Every organisation makes decisions under pressure — about hiring, promotions, performance, discipline, supplier choice, or deploying an AI system. Behavioural science has shown for decades that those decisions are shaped by **predictable cognitive biases** (anchoring, similarity, halo effect, confirmation, availability) and by **systemic biases and noise** baked into processes, rubrics, and algorithms.

Generic LLMs can discuss these problems fluently — but for business-grade decision support they fall short on three things HR, Legal, Risk, and People leaders care about:

1. **Auditability** — where did the analysis come from?
2. **Consistency** — does the same situation get the same analysis tomorrow?
3. **Discipline** — does the model stay inside an established research vocabulary, or invent terms?

This system is designed to address those gaps directly.

---

## How It Works (Technical Summary)

A Retrieval-Augmented Generation (RAG) pipeline with closed-taxonomy constraints and strict structured output:

- **LLM**: Anthropic Claude Sonnet 4.5 via Amazon Bedrock (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`)
- **Embeddings**: Amazon Titan v2 (`amazon.titan-embed-text-v2:0`, 1024 dims, normalised). Chunks use sentence-centred windows (default ±3 sentences per window, mean-pooled to one vector per chunk). User scenarios embed as a single string at query time.
- **Retrieval**: FAISS `IndexFlatIP` over a curated knowledge base, with BM25 hybrid search (RRF fusion), concept and decision-domain filtering, MMR diversification, and an optional LLM-as-judge reranker.
- **Knowledge base**: 300+ chunks across hiring, performance, leadership, legal, AI governance, strategy, and compliance — assembled from seminal decision-science literature.
- **Bias taxonomy**: 60+ entries across cognitive, systemic, and AI-alignment layers.
- **Output**: Strict JSON schema constrained against the taxonomy — auditable and integrable.
- **Hosting**: AWS-native — Bedrock, S3, KMS, CloudWatch, IAM.

For full architecture, pipelines, and operations detail see [`docs/DESIGN.md`](docs/DESIGN.md).

---

## Quickstart

**Prerequisites:** Python 3.10+, an AWS Bedrock-enabled environment, credentials configured in `.env` (see [`.env.example`](.env.example)).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env               # fill in real values
```

Verify connectivity (no UI, just checks Bedrock reachability):

```bash
python -c "from evaluation.connectivity import run_connectivity_check; print(run_connectivity_check()['ok'])"
```

---

## Run the UI

```bash
source .venv/bin/activate
streamlit run ui/app.py
# Opens http://localhost:8501
```

The Streamlit UI provides:
- **Scenario editor** — type a decision situation in plain language, or load one from the built-in catalog
- **Side-by-side comparison** — Baseline (no RAG) vs Knowledge-Enhanced (with RAG) bias analysis
- **Metrics bar** — bias count delta, groundedness score, schema validity, chunk retrieval stats
- **Retrieved chunks panel** — full source attribution per finding
- **JSON export** — download the full `detect_bias_comparison()` result

Sidebar controls let you switch corpus (`public` / `private`), adjust top-k chunks (2–20), and tune MMR lambda (relevance vs diversity).

---

## Common Commands (CLI)

| What you want | Command |
|---|---|
| Run the UI | `streamlit run ui/app.py` |
| One custom scenario | `python scripts/run_my_scenario.py --scenario "…"` or `--file path.txt` |
| Baseline trio from catalog | `python scripts/record_response_run.py --label my_run` |
| Two fixed demo scenarios | `python scripts/run_sample_comparison.py` |
| Full extended suite → JSON report | `python evaluation/run_evaluation.py` |
| Check corpus for new/changed files | `python scripts/check_corpus_updates.py` |
| Re-index after adding new content | `python scripts/check_corpus_updates.py --ingest` |
| Unit tests (no Bedrock) | `python -m unittest discover -s tests -p 'test_*.py' -v` |

---

## Corpus Management

### What the corpus is

The knowledge base is built from source documents in `data/corpora/<corpus>/`. Two corpora ship with the project:

| Corpus | Location | Content |
|---|---|---|
| `public` | `data/corpora/public/` | Synthesised research dossiers (~5–8 KB each) committed to the repo. Safe to use anywhere. |
| `private` | `data/corpora/private/` | Full-book PDFs (gitignored, local-only). Richer retrieval; used for lift studies. |

Set `CORPUS_NAME=public` or `CORPUS_NAME=private` in `.env`, or pass `--corpus` to any script.

### Checking whether the index is up to date

```bash
# Report only — no Bedrock calls, runs in under a second
python scripts/check_corpus_updates.py
python scripts/check_corpus_updates.py --corpus private
```

This compares every file in `parsed_text/` against a SHA-256 fingerprint stored at `data/corpora/<corpus>/corpus_fingerprint.json` and reports:

- 🆕 New files not yet indexed
- ✏️ Changed files (content hash differs from last index)
- 🗑️ Files removed since the last index
- 📄 Raw PDFs in `raw/` that have never been parsed

The Streamlit sidebar shows the same status automatically on every page load (size-based fast check, no hashing).

### Adding a new synthesised dossier (public corpus)

1. Write your `.txt` file and drop it into `data/corpora/public/parsed_text/`
2. Create the matching `.meta.json` sidecar alongside it (copy the format from any existing one and update `source`, `author`, `title`, `publication_year`, `file_name`)
3. Run the pipeline:

```bash
python data_pipeline/build_knowledge_base.py --corpus public
python data_pipeline/build_vector_index.py --corpus public
python scripts/check_corpus_updates.py --corpus public   # updates fingerprint
```

### Adding a new full-book PDF (private corpus)

1. Drop the PDF into `data/corpora/private/raw/`
2. Run the full pipeline (PDF parsing included):

```bash
python data_pipeline/build_knowledge_base.py --corpus private --profile book
python data_pipeline/build_vector_index.py --corpus private
python scripts/check_corpus_updates.py --corpus private   # updates fingerprint
```

### One-command check and rebuild

```bash
# Detects changes and re-runs the full pipeline automatically if anything changed
python scripts/check_corpus_updates.py --ingest

# Force a full rebuild even when the fingerprint reports no changes
python scripts/check_corpus_updates.py --ingest --force
```

### Pipeline stages (reference)

The offline pipeline runs in five stages. `build_knowledge_base.py` runs stages 1–5; `build_vector_index.py` runs stage 6 (requires live Bedrock access for embeddings):

| Stage | Script | Input → Output |
|---|---|---|
| 1. Parse | `data_pipeline/pdf_parser.py` | `raw/*.pdf` → `parsed_text/*.txt` + `.meta.json` |
| 2. Chunk | `data_pipeline/chunker.py` | `parsed_text/*.txt` → `chunks/chunks.json` |
| 3. Tag concepts | `data_pipeline/concept_extractor.py` | `chunks.json` → `chunks_with_concepts.json` |
| 4. Enrich | `data_pipeline/enrich_chunks.py` | `chunks_with_concepts.json` → `processed/enriched/knowledge_base.json` |
| 5. Synthesis | `data_pipeline/synthesis_builder.py` | `knowledge_base.json` → `processed/enriched/synthesis.json` |
| 6. Index | `data_pipeline/build_vector_index.py` | `knowledge_base.json` → `processed/index/` (FAISS + BM25 + metadata) |

After changing only enrichment fields (passage_type, decision_phase) without touching chunk text, skip re-embedding with:

```bash
python data_pipeline/build_vector_index.py --corpus public --metadata-only
```

---

## Scenarios

Scenarios are the evaluation inputs passed to `detect_bias_comparison()`.

| Layer | Path | Purpose |
|---|---|---|
| **Source JSON (edit these)** | `data/eval/gold/baseline_scenarios.json` | 3 frozen baseline scenarios used for A/B comparisons |
| | `data/eval/gold/test_scenarios.json` | Extended test suite |
| | `data/eval/gold/private_book_scenario_questions.json` | 12 questions grounded in specific private-corpus books |
| | `data/eval/gold/gold_labels.json` | Expected bias names and concepts per scenario |
| **Merged catalog (generated)** | `data/eval/gold/scenarios_catalog.json` | Single file the app loads at runtime |

Rebuild the catalog after editing any source JSON:

```bash
python scripts/build_scenarios_catalog.py
```

---

## Repository Map

| Path | What it is |
|---|---|
| `ui/app.py` | Streamlit web UI |
| `app/services/bias_detector.py` | End-to-end orchestration: retrieval + two Claude calls |
| `app/services/bedrock_provider.py` | Bedrock client wrapper (chat + embeddings) |
| `rag/retriever.py` | FAISS + BM25 + RRF + MMR + reranker |
| `rag/fusion.py` | Reciprocal Rank Fusion |
| `rag/hierarchical_retriever.py` | Synthesis context block selection |
| `rag/query_classifier.py` | Intent classification to tune MMR lambda |
| `data_pipeline/` | Offline ingestion pipeline (parse → chunk → tag → enrich → index) |
| `evaluation/` | Metrics, connectivity check, scenario catalog, run cards |
| `shared_components/settings.py` | Single source of truth for all env-var configuration |
| `shared_components/utilities/path_utils.py` | All filesystem path resolution |
| `scripts/check_corpus_updates.py` | Corpus freshness check and incremental ingest trigger |
| `scripts/` | All other operational CLIs |
| `data/corpora/public/` | Public dossier corpus (committed) |
| `data/corpora/private/` | Full-book corpus (gitignored, local-only) |
| `data/eval/gold/` | Scenario catalog and gold labels |
| `data/metadata/bias-taxonomy.json` | 60+ bias vocabulary |
| `data/registry/decision_intelligence_ontology.json` | 8-layer decision-intelligence ontology |
| `prompts/v1/bias_detection_system_prompt.txt` | Versioned system prompt |
| `Dockerfile` | Container build for cloud deployment |
| `docs/DESIGN.md` | Canonical technical design reference |
| `docs/Solution_Design_Document_v4.3_2.docx` | Portfolio solution design (externally maintained) |
| `archived/` | Retired documentation (history and grep reference) |

---

## Deployment

See [`docs/DESIGN.md` §10](docs/DESIGN.md) for the full deployment options table. In brief:

| Target | Cost | Notes |
|---|---|---|
| Local | Free | `streamlit run ui/app.py` + `.env` credentials |
| Streamlit Community Cloud | Free (public) | Push to GitHub, add secrets at share.streamlit.io |
| AWS App Runner | ~$5/month | Docker image on ECR; IAM role for Bedrock — no stored credentials |
| GCP Cloud Run | ~$0–5/month | Same Docker image; inject AWS credentials via Secret Manager |

Container build:

```bash
docker build -t bias-copilot .
docker run -p 8501:8501 --env-file .env bias-copilot
```

---

## Status

- Foundation build complete and runnable
- Public dossier corpus indexed (12 sources, 29 chunks)
- Private full-book corpus available locally for evaluation lift studies
- Streamlit UI live — side-by-side baseline vs RAG comparison with source attribution
- Corpus fingerprinting and incremental ingest check (`scripts/check_corpus_updates.py`)
- Phase 2 (author-published essays in public corpus) — scaffolded, awaiting curation
- Evaluation harness and scenario catalog operational

**Smoke checks:**

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -c "from evaluation.connectivity import run_connectivity_check; print(run_connectivity_check()['ok'])"
```

---

## Disclaimer

This system is **decision support only**. It is **not** legal, financial, medical, or HR advice. Outputs must be reviewed by qualified human decision makers familiar with the local legal, regulatory, and organisational context. The system never automates a decision; it offers a structured, citable second perspective.

---

## Author

**Rommel Sharma** — Enterprise Technology Leader | Applied AI Practitioner

[LinkedIn](https://www.linkedin.com/in/rommelsharma/) · Open to collaboration on decision-intelligence and responsible-AI projects.
