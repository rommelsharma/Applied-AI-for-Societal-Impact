# Design, layout, and operations

Canonical technical reference for **repository layout**, **data paths**, **pipelines**, **runtime behaviour**, **evaluation**, **corpus management**, **UI**, and **deployment**. Narrative deep-dives and historical splits live under [`archived/docs/`](../archived/docs/).

**Binary design artefacts (not overwritten by automation):**
- [`Solution_Design_Document_v4.3_2.docx`](Solution_Design_Document_v4.3_2.docx) — full portfolio solution design including UI and deployment sections.
- [`rag_solution_design_best_practices.docx`](rag_solution_design_best_practices.docx) — RAG design reference.

---

## 1. Purpose

Bias-aware, taxonomy-constrained RAG over a curated decision-science corpus: **baseline vs RAG** comparison on each scenario, **strict JSON** outputs, **source-attributed** retrieval, AWS Bedrock (Claude + Titan). Decision support only — not legal, medical, HR, or financial advice.

**Target operators:** engineers maintaining ingestion and indexes; reviewers auditing eval captures.

---

## 2. Repository layout

```
bias-aware-decision-making-rag-copilot/
├── ui/                        # Streamlit web application
│   └── app.py                 # Single-file UI — wraps detect_bias_comparison()
├── app/services/              # Bedrock provider, bias_detector orchestration
├── rag/                       # KnowledgeRetriever, fusion (RRF), query classifier
├── data_pipeline/             # parse → chunk → concepts → enrich → index
├── evaluation/                # metrics, connectivity, scenario_catalog, run_card
├── shared_components/         # settings, path_utils, taxonomy_utils
├── prompts/v1/                # versioned system prompt (bias_detection_system_prompt.txt)
├── data/
│   ├── registry/              # decision_intelligence_ontology.json
│   ├── metadata/              # bias-taxonomy, retrieval-concepts, book_dossier manifests
│   ├── corpora/
│   │   ├── public/            # committed dossier corpus
│   │   │   ├── raw/           # synthesised source files (gitignored in raw/)
│   │   │   ├── parsed_text/   # .txt + .meta.json per document
│   │   │   ├── chunks/        # intermediate chunk artefacts
│   │   │   ├── processed/
│   │   │   │   ├── enriched/  # knowledge_base.json, synthesis.json
│   │   │   │   └── index/     # FAISS, embeddings.npy, BM25, manifest (v4 layout)
│   │   │   ├── vector_store/  # legacy index path (fallback)
│   │   │   └── corpus_fingerprint.json   # SHA-256 hashes for freshness checks
│   │   └── private/           # full-book PDFs — entire tree gitignored
│   └── eval/
│       ├── gold/              # scenarios_catalog.json + source JSON
│       └── runs/              # eval captures (gitignored; .gitkeep present)
├── scripts/                   # operational CLIs (ingest, index, eval, corpus check)
├── examples/                  # read-only demos and tutorials
├── docs/                      # this file + Word design artefacts
├── Dockerfile                 # container build for cloud deployment
├── .streamlit/                # Streamlit theme and secrets template
├── archived/                  # retired documentation
└── README.md
```

**Path resolution:** `shared_components/utilities/path_utils.py` — corpus-aware; prefers `processed/enriched/` and `processed/index/` (v4), with fallback to legacy `knowledge/` and `vector_store/`.

**`scripts/` vs `examples/`:** `scripts/` holds one-command operational workflows. `examples/` holds read-only demos. New pipeline operations go under `scripts/`; new demos under `examples/`.

---

## 3. Offline pipeline (per corpus)

### Stage order

```
data/corpora/<corpus>/raw/*.pdf
  → pdf_parser.py           → parsed_text/{slug}.txt + .meta.json
  → chunker.py              → chunks/chunks.json
  → concept_extractor.py    → chunks/chunks_with_concepts.json
  → enrich_chunks.py        → processed/enriched/knowledge_base.json
  → synthesis_builder.py    → processed/enriched/synthesis.json
  → build_vector_index.py   → processed/index/ (FAISS, embeddings.npy, BM25, manifest)
```

`build_knowledge_base.py` orchestrates stages 1–5 in sequence. `build_vector_index.py` runs stage 6 separately because it requires live Bedrock access for embeddings.

### Chunking profiles

| Profile | Chunk size | Overlap | When used |
|---|---|---|---|
| `dossier` | 350 words | 60 words | Short synthesised dossiers |
| `book` | 500 words | 80 words | Full-length books |
| `auto` (default) | — | — | Picks `book` for docs > 30 000 words, else `dossier` |

### Chunk embedding

Default: sentence-centred windows per chunk — for each sentence, Titan v2 embeds that sentence ± `RAG_EMBED_SENTENCE_RADIUS` sentences (default ±3), vectors are mean-pooled and L2-normalised → **one vector per chunk row** aligned with `index_metadata.json`. Queries embed the full scenario string (no windowing). Toggle with `RAG_EMBED_SENTENCE_WINDOWS`, `RAG_EMBED_SENTENCE_RADIUS`, `RAG_EMBEDDING_MAX_WINDOWS` in `.env`.

### Metadata-only refresh

When chunk text is unchanged but enrichment fields (passage_type, decision_phase) have been updated, skip re-embedding:

```bash
python data_pipeline/build_vector_index.py --corpus public --metadata-only
```

---

## 4. Corpus management and freshness

### Fingerprint file

`data/corpora/<corpus>/corpus_fingerprint.json` stores a SHA-256 hash, size, and filename for every `.txt` file in `parsed_text/`. Written/updated by `scripts/check_corpus_updates.py` after every successful ingest.

### Checking for updates

```bash
# Report only — no Bedrock calls, safe to run anytime
python scripts/check_corpus_updates.py
python scripts/check_corpus_updates.py --corpus private

# Detect changes and re-run full pipeline automatically
python scripts/check_corpus_updates.py --ingest
python scripts/check_corpus_updates.py --corpus private --ingest

# Force full rebuild even when nothing changed
python scripts/check_corpus_updates.py --ingest --force
```

Exit code 0 = up to date or successfully ingested. Exit code 1 = changes detected but `--ingest` not passed.

### Adding a new synthesised dossier (public corpus)

1. Place `<slug>.txt` and `<slug>.meta.json` in `data/corpora/public/parsed_text/`
2. Run:
```bash
python data_pipeline/build_knowledge_base.py --corpus public
python data_pipeline/build_vector_index.py --corpus public
python scripts/check_corpus_updates.py --corpus public
```

### Adding a new full-book PDF (private corpus)

1. Place the PDF in `data/corpora/private/raw/`
2. Run:
```bash
python data_pipeline/build_knowledge_base.py --corpus private --profile book
python data_pipeline/build_vector_index.py --corpus private
python scripts/check_corpus_updates.py --corpus private
```

### UI freshness indicator

The Streamlit sidebar runs a lightweight size-based check on every page load (no hashing, no Bedrock calls). It shows a green ✅ when the fingerprint is current, or a ⚠️ warning with the exact command to run when new/changed/unparsed files are detected.

---

## 5. Runtime path

1. `concept_extractor` tags the user scenario against taxonomy-derived concepts (and ontology when configured).
2. `KnowledgeRetriever.search` embeds the scenario, runs hybrid retrieval (dense FAISS + optional BM25, RRF fusion), applies MMR, optional overlap-aware filtering, and neighbour expansion.
3. `bias_detector.detect_bias_comparison` builds baseline and RAG system prompts, calls Claude twice, returns `without_rag`, `with_rag`, `retrieval` (traceability rows).

**Key modules:** `app/services/bias_detector.py`, `rag/retriever.py`, `app/services/bedrock_provider.py`.

**UI overrides:** `detect_bias_comparison()` accepts optional `top_k` and `mmr_lambda` kwargs that bypass the frozen `RAG_SETTINGS` dataclass, enabling per-request tuning from the Streamlit sliders without restarting the process.

---

## 6. User interface

The Streamlit application (`ui/app.py`) wraps `detect_bias_comparison()` for non-technical reviewers.

```bash
streamlit run ui/app.py     # http://localhost:8501
```

**Layout:** sidebar (corpus, top-k, MMR lambda, corpus status) · scenario editor (free text or catalog loader) · metrics bar (6 tiles) · side-by-side baseline vs RAG columns · retrieved chunks panel · JSON export.

**Theme:** `.streamlit/config.toml` — professional blue palette, white background.

**Secrets template:** `.streamlit/secrets.toml.example` — for Streamlit Community Cloud deployment.

For full UI design rationale and component descriptions see the *User Interface Design* section of [`Solution_Design_Document_v4.3_2.docx`](Solution_Design_Document_v4.3_2.docx).

---

## 7. Evaluation and scenarios

| Asset / script | Role |
|---|---|
| `data/eval/gold/scenarios_catalog.json` | Canonical **baseline**, **extended_suite**, **private_book_questions**, **gold_expectations**. |
| `evaluation/scenario_catalog.py` | Loaders: `get_baseline_scenarios()`, `get_private_book_questions()`, etc. |
| `scripts/build_scenarios_catalog.py` | Rebuilds catalog from `data/eval/gold/*.json` sources. |
| `evaluation/run_evaluation.py` | Full extended suite → `evaluation/latest_results.json`. |
| `scripts/record_response_run.py` | Baseline slice + metrics + run_card → `data/eval/runs/<ts>_<label>_rag_eval.json`. |
| `scripts/run_sample_comparison.py` | Two fixed scenarios + JSONL append to `data/eval/runs/sample_results_comparison.md`. |
| `examples/run_bias_scenarios_demo.py` | Prints full baseline vs RAG for each extended-suite scenario. |
| `scripts/compare_versions.py` | Multi-corpus comparison; writes under `evaluation/`. |

**Metrics:** schema validity, taxonomy compliance, lexical groundedness (≥0.35 threshold), bias lift (RAG minus baseline count), source diversity, hybrid chunk hits.

**QA (no Bedrock):** `python -m unittest discover -s tests -p 'test_*.py' -v`

**Connectivity smoke:** `python -c "from evaluation.connectivity import run_connectivity_check; print(run_connectivity_check())"`

---

## 8. Deployment

### Local

```bash
source .venv/bin/activate
streamlit run ui/app.py
```

Credentials from `.env` (see `.env.example`). The corpus artefacts live on disk; no separate server process needed.

### Container build

```bash
docker build -t bias-copilot .
docker run -p 8501:8501 --env-file .env bias-copilot
```

`.dockerignore` excludes `.env`, `.venv`, the private corpus, `.claude/` worktrees, and `archived/`.

### Deployment options

| Platform | Cost | Auth model | Best for |
|---|---|---|---|
| **Streamlit Community Cloud** | Free (public) / $25/mo (private) | Secrets panel injects env vars. Rotate bearer token manually. | Reviewer demos, portfolio showcases. |
| **Render / Railway** | $0–7/month | Env vars via dashboard. Docker supported. | Low-cost always-on demos. |
| **GCP Cloud Run** | ~$0–5/month (scale-to-zero) | AWS credentials via Secret Manager. | GCP-first organisations. |
| **AWS App Runner** | ~$5/month | IAM execution role → no stored credentials. | Durable demos; same account as Bedrock. |
| **AWS ECS Fargate + ALB** | $15–40/month | Task execution role; Cognito optional. | Production with HA and access control. |

**Recommended progression:** local → Streamlit Community Cloud → App Runner → ECS Fargate.

**AWS App Runner (ECR push):**

```bash
aws ecr create-repository --repository-name bias-copilot --region us-west-2
docker build -t bias-copilot .
docker tag bias-copilot:latest <account>.dkr.ecr.us-west-2.amazonaws.com/bias-copilot:latest
docker push <account>.dkr.ecr.us-west-2.amazonaws.com/bias-copilot:latest
```

Attach an IAM role with `bedrock:InvokeModel` on the Claude Sonnet and Titan Embed model ARNs. No credentials stored in the image.

**Private corpus in cloud deployments:** the private corpus is excluded from the Docker image by `.dockerignore`. For cloud deployments requiring private-corpus access, mount it from S3 at container start or keep `CORPUS_NAME=public` for cloud and reserve the private corpus for local lift-study evaluation.

---

## 9. Operations

- **Secrets / models:** `.env` from `.env.example` — Bedrock bearer token, region, Sonnet + Titan model IDs, `CORPUS_NAME`, RAG toggles.
- **Corpus selection:** `CORPUS_NAME=public|private` in `.env`, or `--corpus` on any script.
- **Reproducibility:** run cards and manifests capture settings and hashes; keep `scenarios_catalog.json` fixed when A/B testing retrieval flags.
- **Corpus growth:** run `check_corpus_updates.py --ingest` after adding documents. The full pipeline re-runs for all documents in the corpus; incremental embedding of individual files can be added when corpus size warrants it.
- **Production-style concerns** (CI, drift, cost, governance): condensed in [`archived/docs/ml-ops.plan.md`](../archived/docs/ml-ops.plan.md).

---

## 10. Archived documentation

The following files were moved unchanged to `archived/docs/` for history:

- `ARCHITECTURE.md`, `code-flow.md`, `fundamental-concepts.md`, `PROJECT_CONTEXT.md`
- `PRIVATE_CORPUS_GUIDE.md`, `ml-ops.plan.md`, `phased-rollout-plan.md`, `CODEX_HANDBOOK.md`

**Update policy:** change `docs/DESIGN.md` for the live system contract; touch `archived/docs/*` only when preserving an intentional historical snapshot.
