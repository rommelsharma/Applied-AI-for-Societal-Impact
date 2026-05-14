# ML-Ops Plan

This document describes the ML-Ops practices used (and planned) for the Bias-Aware Decision Making RAG Copilot. It aligns with the ML-Ops narrative in the validated portfolio SDD ([`Solution_Design_Document.docx`](Solution_Design_Document.docx), externally maintained — not overwritten by repo scripts).

The system is small (one runtime path, two corpora, one chat model, one embedding model) but the operational practices below are deliberately enterprise-shaped so the same patterns scale when the corpus, traffic, or team grow.

---

## 1. Why a RAG-Specific ML-Ops Plan

A RAG system has three operational artefacts that need independent lifecycles:

| Artefact | Owner of change | Cadence |
|---|---|---|
| **Code** (parser, chunker, retriever, prompt builder) | Engineering | Per pull request |
| **Corpus** (raw PDFs → parsed → chunks → knowledge base → vector index) | Curators / domain experts | Per content addition |
| **Foundation models** (Claude Sonnet 4.5, Titan v2) | Vendor (AWS / Anthropic) | Provider releases |

Conventional MLOps assumes a *trained* model. Here the LLMs are pre-trained and the operational lever is the corpus and the prompt. The plan below treats the corpus as a first-class artefact with the same versioning, evaluation, and deployment discipline normally applied to model weights.

---

## 2. Reproducibility

### 2.1 Environment

- `requirements.txt` pins the floors for `boto3`, `faiss-cpu`, `numpy`, `PyMuPDF`, `python-dotenv`.
- `.env.example` declares every environment variable the runtime reads. `.env` is gitignored.
- `shared_components/settings.py` is the **only** module that reads environment variables; every other file imports `BEDROCK_SETTINGS` and `RAG_SETTINGS`.
- `shared_components/utilities/path_utils.py` is the **only** module that resolves filesystem paths.

### 2.2 Determinism

- Chunking parameters (`PROFILES`, `_FRONT_HEADING`, `_BACK_HEADING`, `_TRIM_SAFETY_THRESHOLD`) are module-level constants.
- The taxonomy (`bias-taxonomy.json`) and support-concepts catalog (`retrieval-concepts.json`) are committed JSON files.
- Bedrock chat is invoked at temperature `0.1` so the same scenario produces near-identical output across runs.
- Bedrock Titan v2 embeddings are deterministic for a given **input string**; the **index vector** for a chunk is the pooled result of **all chosen sentence-window strings**, so it changes if `RAG_EMBED_SENTENCE_RADIUS`, `RAG_EMBEDDING_MAX_WINDOWS`, or `RAG_EMBED_SENTENCE_WINDOWS` changes.
- FAISS `IndexFlatIP` returns ranked candidates in stable order for a given query vector.

### 2.3 Build manifests

Every vector index writes a `manifest.json` next to the FAISS file:

```json
{
  "corpus": "public",
  "count": 301,
  "dimension": 1024,
  "index_type": "IndexFlatIP",
  "chunking_profile_distribution": {"dossier": 29, "book": 272},
  "embedding_sentence_windows": true,
  "embedding_sentence_radius": 3,
  "embedding_max_windows_per_chunk": 0
}
```

The runtime can refuse to start when the corpus name, chunk count, or dimension does not match expectations from configuration.

---

## 3. Versioning

The following artefacts move on independent clocks. Each gets an explicit version handle.

| Versioned thing | Where it lives | How it advances |
|---|---|---|
| Code | git commit hash | Per pull request |
| Bedrock chat model | `BEDROCK_MODEL_ID` in `.env` | Manual env change + redeploy |
| Bedrock embedding model | `BEDROCK_EMBEDDING_MODEL_ID` in `.env` | Forces a full re-embed of the corpus |
| Bias taxonomy | `data/metadata/bias-taxonomy.json` (committed) | PR with reviewer sign-off |
| Corpus snapshot | `data/corpora/<name>/vector_store/manifest.json` + git tag of the build commit | Per ingestion run |
| Embedding **window** policy | `RAG_EMBED_SENTENCE_WINDOWS`, `RAG_EMBED_SENTENCE_RADIUS`, `RAG_EMBEDDING_MAX_WINDOWS` in `.env` (mirrored in `manifest.json`) | Any change requires **`build_vector_index.py` re-run** (invalidates vectors like an embedding-model swap) |

A change to the embedding model id is the most disruptive and is flagged as such in code review: it invalidates every existing `embeddings.npy`.

---

## 4. Data and Model Lineage

Every chunk in the knowledge base carries a complete chain of custody:

```text
raw PDF
  ├─ source_filename, sha256 (planned)
  └─ parsed_text + .meta.json (parser_notes for any field that could not be inferred)
       └─ chunks.json (cleaning_notes, chunking_profile, chapter_title)
            └─ chunks_with_concepts.json (concept tags, concept_confidence)
                 └─ knowledge_base.json (passage_type, decision_phase, classifier, importance, decision_domains, summary, keywords)
                      └─ embeddings.npy + index_metadata.json (1:1 row alignment; manifest records embed model + sentence-window pooling policy)
                           └─ retrieval result (similarity score, chunk id, source, chapter, concepts, passage_type)
                                └─ bias_detector response payload (`retrieval` array preserved alongside `with_rag` JSON)
```

This contract is what allows the runtime payload to point at *exactly* which chunk shaped which answer. Adding `sha256` of each raw PDF to the parser sidecar is an explicit short-term enhancement.

---

## 5. CI/CD Strategy

### 5.1 Code pipeline (continuous integration)

A typical pull request runs through this pipeline:

```text
[1] lint          : ruff / black formatting
[2] unit tests    : module-level tests for chunker, concept extractor, MMR, retriever fallbacks
[3] schema tests  : taxonomy and support-concepts JSON validate against an explicit JSON schema
[4] smoke ingest  : a tiny synthetic PDF flows through parse → chunk → enrich without external calls
[5] container     : docker build, image-scan via Trivy, push to ECR on merge
```

### 5.2 Corpus pipeline (continuous corpus delivery)

A change to taxonomies, dossier manifests, or PDFs runs:

```text
[1] schema validate     : every JSON file in data/metadata/ matches its schema
[2] dossier rebuild     : `python scripts/generate_book_dossiers.py` regenerates dossier PDFs deterministically
[3] knowledge build     : `python data_pipeline/build_knowledge_base.py --corpus public`
[4] vector index build  : `python data_pipeline/build_vector_index.py --corpus public`
[5] evaluation harness  : `python evaluation/run_evaluation.py` runs the canonical scenarios
[6] gating              : evaluation report is compared to the previous baseline; regressions block merge
[7] artefact upload     : new vector store uploaded to S3 with a versioned key
```

The private corpus pipeline is identical but runs only on the operator's local laptop and writes nowhere outside `data/corpora/private/`.

### 5.3 Deployment

- **Blue/green** deployment via ECS service-update or Lambda alias-shift on traffic.
- **Manual approval gate** for the production environment until automated evaluation gating reaches the agreed quality bar.
- **Vector-store hot-reload** (planned): the runtime watches a versioned S3 prefix and swaps to the new manifest atomically, so corpus refreshes do not require a redeploy.

---

## 6. Evaluation Gating

Beyond unit tests, the deployment pipeline runs `evaluation/run_evaluation.py` against the canonical 30-plus scenarios on every release candidate. Releases are blocked when:

- **Schema fidelity** drops below 100% — the model's output must parse against the locked JSON schema.
- **Taxonomy adherence** drops below 100% — any `bias_name` outside `bias-taxonomy.json` is a hard fail.
- **Bias-count delta** between baseline and RAG indicates a regression. Specifically: the RAG path should never identify *fewer* biases on average than baseline on a given evaluation cut.
- **P95 inference latency** exceeds the agreed SLO (currently 8 seconds end-to-end).
- **Empty-retrieval fraction** exceeds a small threshold (currently 1%) — any spike indicates corpus or query encoder drift.

A nightly job runs `scripts/compare_versions.py --versions public,private` so the lift between corpora is monitored as a moving baseline rather than a one-off measurement.

---

## 7. Observability

### 7.1 Structured logs

Every retrieval and inference call emits a JSON log line with:

```text
trace_id, scenario_id, corpus, top_k, mmr_lambda, retrieved_chunk_ids,
similarity_scores, baseline_call_ms, rag_call_ms, json_parse_retries,
input_tokens, output_tokens, model_id
```

CloudWatch logs are KMS-encrypted; scenario text is not logged in clear unless the operator explicitly enables debug verbosity.

### 7.2 Metrics (CloudWatch / OpenTelemetry)

- P50 / P95 retrieval latency.
- P50 / P95 inference latency, baseline and RAG paths.
- JSON-parse retry rate.
- Empty-result rate (after concept/domain filter, before fallback).
- Bedrock throttling rate.
- Cost per scenario (estimated from input/output token counts).

### 7.3 Traces

Distributed tracing (AWS X-Ray) wraps each request in spans for `embed_query`, `faiss_search`, `mmr`, `rerank` (when enabled), `format_context`, `baseline_call`, and `rag_call`. The trace_id is logged so a single CloudWatch query reaches the full timeline.

### 7.4 Alarms

- Bedrock throttling rate > 1% over 5 minutes.
- JSON-parse retry rate > 5%.
- Empty-result rate > 5%.
- P95 inference latency > 12 seconds.
- Sudden change (> 30%) in average bias count surfaced by the RAG path versus the trailing 7-day baseline.

---

## 8. Cost Management

Bedrock cost is the dominant operating expense. The plan controls it through five levers:

1. **Cache pooled chunk vectors.** At index build, each chunk produces **one** row in `embeddings.npy` (sentence-window Titan calls are pooled offline). Queries embed the scenario once per request — no per-token chat cost for retrieval.
2. **Two-layer retrieval.** FAISS returns a wide candidate pool and the optional reranker is invoked only on a fixed top-N (default 24), not on the full corpus.
3. **MMR before reranker.** When the optional reranker is enabled it runs after MMR, so the LLM scores at most a constant number of candidates.
4. **Off-by-default LLM enrichment.** `passage_type` and `decision_phase` are derived rule-based for free; the LLM-enrichment path (`ENRICHMENT_USE_LLM=true`) is opt-in.
5. **Off-by-default reranker.** `RAG_RERANKER=none` is the default. The Claude-Haiku reranker is documented but not enabled until corpus size warrants it (~3,000 chunks).

A monthly cost dashboard (CloudWatch + Cost Explorer tags) splits spend by `chat`, `embedding-runtime`, `embedding-corpus-build`, and `reranker`.

---

## 9. Security and Privacy

| Concern | Practice |
|---|---|
| Inbound traffic | HTTPS-only via ALB / API Gateway with ACM certificates |
| Bedrock auth | Bearer token in AWS Secrets Manager, rotated quarterly |
| Encryption at rest | KMS on S3 (vector store), Secrets Manager, EBS, CloudWatch logs |
| Encryption in transit | TLS 1.2+ everywhere; Bedrock VPC endpoint avoids public internet |
| IAM | Least-privilege roles separated by function: `corpus-ingest`, `runtime-inference`, `eval-runner`, `admin` |
| Audit | CloudTrail for every Bedrock call; structured logs preserved for the agreed retention period |
| Data minimisation | Scenarios redacted in logs by default; opt-in debug only in dev |
| Provider data use | Bedrock does not use prompts to train foundation models — confirmed contractually |
| Private corpus | `data/corpora/private/` is gitignored and never deployed; private vector stores remain on the operator's laptop |

---

## 10. Responsible AI

- **Disclaimer surfaced in every UI and in the response payload**: not legal/medical/HR/financial advice.
- **Closed-vocabulary output.** The taxonomy is injected into the system prompt and the LLM is constrained to use only its named bias entries.
- **Source attribution.** Every RAG response carries a `retrieval` array with chunk ids, sources, similarity scores, and concepts.
- **Human-in-the-loop is the operating model.** The system surfaces analysis and recommended actions; humans make the decision.
- **Coverage and fairness.** The corpus's domain coverage (HR, legal, AI governance, leadership, strategy, compliance) and known limitations (English-language, Western-canonical bias) are documented in the README so consumers understand the lens.
- **Audit-ready history.** Scenarios, retrieved chunks, model ids, and final outputs are retained under data-handling controls so any answer can be reconstructed and reviewed.

---

## 11. Drift Monitoring

Three independent drift surfaces are watched:

1. **Corpus drift.** When chunk counts or concept distributions shift more than ±10% between two builds of the same corpus, a review job opens an issue.
2. **Retrieval-quality drift.** The same canonical query suite runs nightly; significant changes in average similarity score or in the set of top-k chunks for a scenario flag a regression.
3. **Output drift.** The bias-count distribution and recommended-action count distribution per domain are tracked; abrupt changes are escalated for review (a model-vendor upgrade often presents this way).

---

## 12. Disaster Recovery

| Artefact | Backup | RPO / RTO target |
|---|---|---|
| Vector store (public) | Versioned S3 bucket with object-lock | RPO 24h / RTO 1h |
| Taxonomies and prompts | git history | RPO 0 / RTO < 15min |
| Container images | ECR with lifecycle policy retaining last 30 images | RPO 0 / RTO < 30min |
| Secrets | AWS Secrets Manager rotation history | RPO 0 / RTO < 30min |
| Private corpus | Local-laptop backup (Time Machine / equivalent) — not in scope for production DR | n/a |

A failed Bedrock region is mitigated by Bedrock cross-region failover (AWS-managed) and, for a future hardening, a secondary Bedrock region in `us-east-1` configured as warm standby.

---

## 13. Roadmap For This Plan

| Item | Priority | Status |
|---|---|---|
| sha256 of raw PDFs in parser sidecars | Medium | Not started |
| LLM-as-judge grounding evaluator on top of `compare_versions.py` | Medium | Not started |
| S3-backed vector store hot reload | High | Designed, not implemented |
| Container + Fargate deployment with Bedrock VPC endpoint | High | Designed, not implemented |
| OpenTelemetry SDK integration (replaces ad-hoc structured logs) | Medium | Not started |
| Cost dashboard split by `chat / embed-runtime / embed-build / rerank` | Low | Not started |
| Model-card publishing (per corpus, per release) | Medium | Designed, not implemented |
| Automated taxonomy expansion review workflow (PR + reviewer + diff in concept distribution) | Low | Not started |

---

## 14. Operating Conventions

- **Two corpora coexist.** `public` ships in the demo and the design document; `private` runs only locally and feeds `compare_versions.py`. Public-facing reports always state which corpus produced the numbers.
- **Configuration is data, not code.** Every prompt template, taxonomy entry, and chunking parameter lives in a file someone can read without opening Python.
- **Failure is logged, not crashed.** Reranker errors, malformed JSON, and missing parser metadata all degrade gracefully and emit a note rather than aborting the request.
- **Comparisons replace assertions.** Whenever the team is tempted to claim "RAG helps here," `compare_versions.py` is the right tool to produce evidence rather than belief.
