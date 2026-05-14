# Design, layout, and operations

Single canonical reference for **repository layout**, **data paths**, **pipelines**, **runtime behaviour**, **evaluation**, and **day-two operations**. Narrative deep-dives and historical splits live under [`archived/docs/`](../archived/docs/) (same filenames as before consolidation).

**Binary references (not overwritten by automation):**

- [`Solution_Design_Document.docx`](Solution_Design_Document.docx) — portfolio solution design.
- [`rag_solution_design_best_practices.docx`](rag_solution_design_best_practices.docx) — RAG design notes.

---

## 1. Purpose

Bias-aware, taxonomy-constrained RAG over a curated decision-science corpus: **baseline vs RAG** comparison on each scenario, **strict JSON** outputs, **source-attributed** retrieval, AWS Bedrock (Claude + Titan). Decision support only — not legal, medical, HR, or financial advice.

**Target operators:** engineers maintaining ingestion and indexes; reviewers auditing eval captures.

---

## 2. Top-level layout

```text
bias-aware-decision-making-rag-copilot/
├── app/services/              # Bedrock provider, bias_detector orchestration
├── rag/                       # KnowledgeRetriever, fusion (e.g. RRF), hybrid search
├── data_pipeline/             # parse → chunk → concepts → enrich → index
├── evaluation/                # metrics, connectivity, scenario_catalog, run_card
├── shared_components/         # settings, path_utils, taxonomy_utils
├── prompts/v1/                # versioned system prompt (bias_detection_system_prompt.txt)
├── data/
│   ├── registry/              # decision_intelligence_ontology.json
│   ├── metadata/              # bias-taxonomy, retrieval-concepts, book_dossiers manifests
│   ├── corpora/<corpus>/      # public | private — raw, parsed_text, chunks, …
│   └── eval/
│       ├── gold/              # scenarios_catalog.json + source JSON (baseline, test, private_book, gold_labels)
│       └── runs/              # eval captures (tracked empty via .gitkeep; generated files gitignored)
├── scripts/                   # operational CLIs: ingest, build index, eval, compare, record runs
├── examples/                  # read-only demos / tutorials (how to call the stack from Python or the terminal)
├── docs/                      # this file + Word design artefacts
├── archived/                  # legacy captures, optional fallbacks, superseded markdown
└── README.md
```

**Path resolution:** `shared_components/utilities/path_utils.py` — corpus-aware; prefer `processed/enriched/` and `processed/index/` (v4), with fallback to legacy `knowledge/` and `vector_store/`.

**`scripts/` vs `examples/`:** `scripts/` holds one-command operational workflows (ingest, build index, rebuild catalog, record runs, comparisons). `examples/` holds demos and tutorials that show how to use the system without being part of the maintenance toolchain. New pipeline operations go under `scripts/`; new demos go under `examples/`.

---

## 3. Offline pipeline (per corpus)

Typical flow (see `data_pipeline/build_knowledge_base.py` and `build_vector_index.py`):

```text
data/corpora/<corpus>/raw/*.pdf
  → pdf_parser.py          → parsed_text/{slug}.txt + .meta.json
  → chunker.py             → chunks/chunks.json
  → concept_extractor.py   → chunks/chunks_with_concepts.json
  → enrich_chunks.py       → processed/enriched/knowledge_base.json
  → synthesis_builder.py   → processed/enriched/synthesis.json (when used)
  → build_vector_index.py  → processed/index/ (FAISS, embeddings.npy, manifest, BM25 when built)
```

**Chunk embedding (default):** sentence-centred windows per chunk, Titan v2 per window, mean-pool + L2-normalise → **one vector per chunk row** aligned with `index_metadata.json`. Env toggles: `RAG_EMBED_SENTENCE_WINDOWS`, `RAG_EMBED_SENTENCE_RADIUS`, `RAG_EMBEDDING_MAX_WINDOWS` (see `.env.example`). **Queries** embed the full scenario string (no sentence windows).

**Corpus selection:** `CORPUS_NAME` env or `--corpus` on scripts.

---

## 4. Runtime path

1. `concept_extractor` tags the user scenario against taxonomy-derived concepts (and ontology when configured).
2. `KnowledgeRetriever.search` embeds the scenario, hybrid retrieval (vector + optional BM25, fusion), MMR, optional overlap-aware filtering and neighbour expansion (`RAG_OVERLAP_*`, `RAG_CONTEXT_EXPAND_*` in settings).
3. `bias_detector.detect_bias_comparison` builds baseline and RAG system prompts, calls Claude twice, returns `without_rag`, `with_rag`, `retrieval` (traceability rows).

**Key modules:** `app/services/bias_detector.py`, `rag/retriever.py`, `app/services/bedrock_provider.py`.

---

## 5. Evaluation and scenarios

**Operator quickstart:** step-by-step commands, scenario file locations, and the rebuild catalog command are in **[`README.md`](../README.md)** (sections *Quickstart: run the code* and *Scenarios: where they live and how to change them*). This section lists the canonical **artefacts** and **modules**.

| Asset / script | Role |
|----------------|------|
| `data/eval/gold/scenarios_catalog.json` | Canonical **baseline**, **extended_suite**, **private_book_questions**, **gold_expectations**. |
| `evaluation/scenario_catalog.py` | Loaders: `get_baseline_scenarios()`, `get_extended_test_scenarios()`, etc. |
| `scripts/build_scenarios_catalog.py` | Rebuilds catalog from ``data/eval/gold/*.json`` sources; ``evaluation/`` (and an optional local ``archived/evaluation_jsonsources/`` folder if you recreate it) are fallbacks only. |
| `evaluation/run_evaluation.py` | Full extended suite → `evaluation/latest_results.json`. |
| `scripts/record_response_run.py` | Baseline slice + metrics + run_card → `data/eval/runs/<ts>_<label>_rag_eval.json`. |
| `scripts/run_sample_comparison.py` | Two fixed scenarios + JSONL append to `data/eval/runs/sample_results_comparison.md`. |
| `examples/run_bias_scenarios_demo.py` | Prints full baseline vs RAG for each extended-suite scenario (IDE / terminal; not an `.ipynb`). |
| `scripts/compare_versions.py` | Multi-corpus comparison; writes under `evaluation/`. |

**Layout:** ``evaluation/`` is the **library** (metrics, connectivity, scenario catalog, run cards). ``scripts/`` are **CLI runners** that import it and call Bedrock for captures.

**QA (no Bedrock):** `python -m unittest discover -s tests -p 'test_*.py' -v`

**Connectivity smoke:** `python -c "from evaluation.connectivity import run_connectivity_check; print(run_connectivity_check())"` — logs to `data/eval/runs/connectivity_log.txt`.

---

## 6. Private corpus (local-only)

- **Location:** `data/corpora/private/` — entire tree **gitignored**; never commit full books.
- **Build:** `python data_pipeline/build_knowledge_base.py --corpus private --profile book` then `python data_pipeline/build_vector_index.py --corpus private`.
- **Compare to public:** `scripts/compare_versions.py --versions public,private` (markdown summary avoids leaking private text).

Full operator detail: [`archived/docs/PRIVATE_CORPUS_GUIDE.md`](../archived/docs/PRIVATE_CORPUS_GUIDE.md).

---

## 7. Operations and rollout (summary)

- **Secrets / models:** `.env` from `.env.example` — Bedrock bearer token, region, Sonnet + Titan IDs, `CORPUS_NAME`, RAG toggles.
- **Reproducibility:** run cards and manifests capture settings and hashes; keep `scenarios_catalog.json` fixed when A/B testing retrieval flags.
- **Corpus growth:** phased addition of dossiers and optional author content (`scripts/ingest_author_content.py`) — historical plan text: [`archived/docs/phased-rollout-plan.md`](../archived/docs/phased-rollout-plan.md).
- **Production-style concerns** (CI, drift, cost, governance): condensed narrative in [`archived/docs/ml-ops.plan.md`](../archived/docs/ml-ops.plan.md).

---

## 8. Archived documentation

The following Markdown files were **moved unchanged** to `archived/docs/` for history and grep-friendly deep dives:

- `ARCHITECTURE.md`, `code-flow.md`, `fundamental-concepts.md`, `PROJECT_CONTEXT.md`
- `PRIVATE_CORPUS_GUIDE.md`, `ml-ops.plan.md`, `phased-rollout-plan.md`, `CODEX_HANDBOOK.md`

**Update policy:** change **`docs/DESIGN.md`** for the live system contract; touch `archived/docs/*` only when preserving an intentional historical snapshot.
