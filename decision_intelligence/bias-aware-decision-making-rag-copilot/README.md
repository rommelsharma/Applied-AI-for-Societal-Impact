# Bias-Aware Decision Making RAG Copilot

**A research-grounded AI assistant that helps people, teams, and HR functions make fairer, more consistent decisions — by surfacing the cognitive and systemic biases hiding inside everyday business situations.**

It is built on top of large language models (LLMs) but adds the rails that decision-grade work needs: a curated knowledge base of seminal decision-science research, a closed and auditable bias vocabulary, source attribution for every finding, and a strict, structured output that integrates cleanly into existing review and case-management workflows.

> Decision support only. Not legal, medical, HR, or financial advice. Always pair with human judgment and domain expertise.

---

## Why This Exists

Every organisation makes decisions under pressure — about hiring, promotions, performance, discipline, supplier choice, sentencing, or deploying an AI system. Behavioural science has shown for decades that those decisions are **shaped by predictable cognitive biases** (anchoring, similarity, halo effect, confirmation, availability) and by **systemic biases and noise** baked into processes, rubrics, and algorithms.

Generic LLMs (ChatGPT, Claude, Gemini, etc.) can talk about these problems fluently — but for business-grade decision support they fall short on three things HR, Legal, Risk, and People leaders care about:

1. **Auditability** — where did the analysis come from?
2. **Consistency** — does the same situation get the same analysis tomorrow?
3. **Discipline** — does the model stay inside an established research vocabulary, or invent terms?

This system is designed to address those gaps directly.

---

## What It Does, In Plain Language

You describe a real situation in everyday language — for example, *"a senior manager is choosing between two internal candidates and is privately worried that the stronger candidate is a threat to his own promotion"* — and the system returns:

- A **situation summary**.
- The **cognitive biases** at play (e.g. similarity bias, self-serving bias, in-group bias).
- The **systemic biases and noise** at play (e.g. process ambiguity, missing rubric, level noise across reviewers).
- The **risks** if the decision is left as is.
- **Recommended actions**, each grounded in established practice (structured interviews, decision-quality scorecards, premortems, mandatory dissent, calibrated rating scales, etc.).
- **Alternative perspectives** that a careful reviewer would consider.
- The **literature it drew on** for each finding — by source, author, and concept tag.

Every analysis is returned in a strict, repeatable structure — easy to read, easy to integrate into HRIS, ATS, case-management, or governance review tools.

---

## How This Stands Apart From Asking a Generic LLM Directly

| Concern | Generic LLM | Bias-Aware Decision Copilot |
|---|---|---|
| **Bias names used** | Free-form; may invent or paraphrase terms | Constrained to a closed taxonomy of 60+ established biases (cognitive, systemic, AI-alignment) |
| **Citations** | Often missing or hallucinated | Every finding cites the chunk, source, author, and concept it came from |
| **Knowledge base** | Whatever happens to be in training data | Hand-curated knowledge base of seminal decision-science books and research |
| **Reproducibility** | Different answer each run | Temperature-locked deterministic runs; identical inputs produce near-identical outputs |
| **Auditability** | Black box | Full chain of custody — raw source → chunk → retrieval → context → answer |
| **Output shape** | Free text; messy to integrate | Strict JSON schema for direct integration into HRIS, ATS, case-management, governance tooling |
| **Bias coverage** | Mostly cognitive (well-known names) | Cognitive **and** systemic **and** AI-governance layers |
| **Disclaimer surfacing** | Inconsistent | Built into every output — never positions itself as legal, medical, or HR advice |
| **Updateability** | New training run required | Knowledge base refreshed without retraining the LLM |
| **Comparison built-in** | None | Can run the **same scenario with and without the curated knowledge base** to demonstrate the lift |

In short — generic LLMs are great conversation partners. This system is built to be a **reviewable, citable, governance-friendly second pair of eyes** that an HR, Legal, Risk, or People function can defend in front of a regulator, a board, or a tribunal.

---

## Where It Adds Value

This isn't a tool for everyday questions. It is designed for the moments where a decision **really matters**:

- **Hiring committees** that want to interrogate a "culture fit" call before it lands.
- **Performance review calibration** across teams and reviewers.
- **Promotion and succession decisions**, especially where lived experience differs from data.
- **Discipline, grievance, and exit decisions** with material legal or reputational exposure.
- **Algorithmic decision review** — interrogating an HR analytics, scoring, or risk model before it goes live.
- **Strategic decisions** with high stakes (M&A, supplier choice, market entry).
- **Legal case preparation** — surfacing anchoring, base-rate neglect, and availability effects in evidence handling.
- **AI governance reviews** — applying decision-science discipline to model deployment choices, alignment risks, and human-in-the-loop design.

---

## Quick Example (Illustrative Output)

For the hiring scenario above, the system surfaces (excerpt, abbreviated):

> **Biases identified**
>  - `similarity_bias` (high confidence) — Candidate B shares the manager's university, hobbies, and social circle; the manager weighs these signals over performance evidence.
>  - `self_serving_bias` (high confidence) — The manager privately frames the stronger candidate as "a threat to his own promotion."
>  - `rationalization_bias` (high confidence) — The decision is justified to HR as "culture and team fit" with no scoring rubric.
>  - `process_ambiguity_bias` (systemic, high) — Absence of a structured comparison or documentation creates space for affinity-driven choice.
>
> **Recommended actions** (excerpt)
>  - Require a behaviorally-anchored rating scale and independent scoring before any "culture fit" framing is applied.
>  - Apply a structured-interviewing rubric (per Kahneman, Sibony, Sunstein, *Noise*).
>  - Re-run the comparison with reviewer-blind scoring before HR endorses the decision.
>
> **Sources retrieved**
>  - *Noise: A Flaw in Human Judgment* — Kahneman, Sibony, Sunstein (5 chunks; concepts: structured interviewing, decision hygiene, similarity bias)

A full side-by-side baseline-vs-this-system comparison on real scenarios is appended to [`data/eval/runs/sample_results_comparison.md`](data/eval/runs/sample_results_comparison.md) when you run `scripts/run_sample_comparison.py` (JSONL rows after a short header). A legacy narrative copy may have been migrated from `docs/sample_results_comparison.md` once.

---

## How It Works (Brief Technical Summary)

The system is a Retrieval-Augmented Generation (RAG) pipeline with closed-taxonomy constraints and strict structured output:

- **LLM**: Anthropic Claude Sonnet 4.5 via Amazon Bedrock (`us.anthropic.claude-sonnet-4-5-20250929-v1:0`).
- **Embeddings**: Amazon Titan v2 (`amazon.titan-embed-text-v2:0`, 1024 dims, normalised). **Chunks** use **sentence-centred windows** (default ±3 sentences per window, mean-pooled to one vector per chunk at index build; configurable via `RAG_EMBED_*` / `RAG_EMBEDDING_MAX_WINDOWS` in `.env`). **User scenarios** embed as a **single** string at query time.
- **Retrieval**: FAISS `IndexFlatIP` over a curated knowledge base, with concept and decision-domain filtering, MMR diversification, and an optional LLM-as-judge reranker.
- **Knowledge base**: 300+ chunks today across hiring, performance, leadership, legal, AI governance, strategy, and compliance contexts — assembled from seminal decision-science literature.
- **Bias taxonomy**: 60+ entries across cognitive biases (anchoring, halo, similarity, etc.), systemic biases (process ambiguity, level noise, algorithmic, historical), and AI-alignment biases (instrumental convergence, specification gaming, etc.).
- **Output**: Strict JSON schema constrained against the taxonomy — predictable, auditable, integrable.
- **Hosting**: AWS-native — Bedrock, S3, KMS, CloudWatch, IAM.

For architecture, paths, pipelines, evaluation, and operations in one place, see [`docs/DESIGN.md`](docs/DESIGN.md). The validated portfolio **Solution Design** is [`docs/Solution_Design_Document.docx`](docs/Solution_Design_Document.docx) (externally maintained; do not overwrite with repo scripts). Superseded long-form Markdown lives under [`archived/docs/`](archived/docs/).

---

## Quickstart: run the code

**Prerequisites:** Python 3.10+, an AWS Bedrock–enabled environment, and credentials (see [`.env.example`](.env.example): bearer token or IAM, region, model IDs).

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env               # then fill in real values
python -c "from evaluation.connectivity import run_connectivity_check; print(run_connectivity_check()['ok'])"
```

**Common commands**

| What you want | Command |
|----------------|---------|
| **One custom scenario** (CLI, file, or stdin) | `python scripts/run_my_scenario.py --scenario "…"` or `--file path.txt` — writes `data/eval/runs/my_scenario_result.json` by default. |
| **Frozen baseline trio** (from the catalog) | `python scripts/record_response_run.py --label my_run` — writes `data/eval/runs/<timestamp>_my_run_rag_eval.json` and can append `sample_results_comparison.md`. |
| **Same baseline, your own JSON list** | `python scripts/record_response_run.py --label x --scenarios path/to/scenarios.json` (array of objects with at least `scenario` text; optional `id`, `title`, `domain`). |
| **Two fixed demo scenarios** (hardcoded in script) | `python scripts/run_sample_comparison.py` — edits live in `scripts/run_sample_comparison.py` (`SCENARIOS`). |
| **Print every extended-suite scenario** (verbose) | `python examples/run_bias_scenarios_demo.py` — reads **`extended_suite`** from the catalog. |
| **Full extended suite → JSON report** | `python evaluation/run_evaluation.py` — reads **`extended_suite`** from the catalog; writes `evaluation/latest_results.json` (gitignored by default). |
| **Unit tests (no Bedrock)** | `python -m unittest discover -s tests -p 'test_*.py' -v` |

**Corpus:** set `CORPUS_NAME=public` or `private` in `.env` (or pass `corpus=` where the API allows). Ingestion/index scripts accept `--corpus`.

---

## Scenarios: where they live and how to change them

| Layer | Path | Purpose |
|--------|------|--------|
| **Source JSON (edit these)** | [`data/eval/gold/baseline_scenarios.json`](data/eval/gold/baseline_scenarios.json), [`test_scenarios.json`](data/eval/gold/test_scenarios.json), [`private_book_scenario_questions.json`](data/eval/gold/private_book_scenario_questions.json), [`gold_labels.json`](data/eval/gold/gold_labels.json) | Human-edited scenario definitions and gold hints. |
| **Merged catalog (generated)** | [`data/eval/gold/scenarios_catalog.json`](data/eval/gold/scenarios_catalog.json) | Single file the app loads at runtime (`evaluation/scenario_catalog.py`). Rebuild after editing sources. |

**Rebuild the catalog** after you change any source JSON:

```bash
python scripts/build_scenarios_catalog.py
```

**In code**, use `get_baseline_scenarios()`, `get_extended_test_scenarios()`, `get_private_book_questions()`, and `scenario_text_for_detection()` from [`evaluation/scenario_catalog.py`](evaluation/scenario_catalog.py).

**Not** in the gold files: `scripts/run_sample_comparison.py` embeds its two demo scenarios in the `SCENARIOS` list at the top of that file — change that list only for those ad-hoc demos.

---

## What's In The Repository

| Document | What it covers |
|---|---|
| [`README.md`](README.md) | **Start here:** quickstart (venv, install, connectivity), commands to run analyses, and **where scenario JSON lives** vs the merged catalog. |
| [`docs/DESIGN.md`](docs/DESIGN.md) | **Canonical** technical design: layout, v4 data paths, offline + runtime pipelines, eval artefacts table, private corpus summary, operations pointers. |
| [`docs/Solution_Design_Document.docx`](docs/Solution_Design_Document.docx) | **Canonical** portfolio solution design (externally authored). **Do not overwrite** from repository automation. |
| [`docs/rag_solution_design_best_practices.docx`](docs/rag_solution_design_best_practices.docx) | RAG design reference (binary). |
| [`data/eval/runs/sample_results_comparison.md`](data/eval/runs/sample_results_comparison.md) | Append-only JSONL log (`scripts/run_sample_comparison.py` or `scripts/record_response_run.py`). |
| [`archived/docs/`](archived/docs/) | Retired splits (`ARCHITECTURE.md`, `code-flow.md`, `fundamental-concepts.md`, rollout and ML-ops plans, private corpus guide, Codex handbook). |

The `data/corpora/public/raw/` folder holds the **synthesised, project-authored research dossiers** that ship with the public repository (~5–8 KB each). **Full source books are never committed** — they live only in the private, gitignored research corpus on the operator's machine.

---

## Status

- Foundation build complete and runnable.
- Public dossier corpus indexed for retrieval (see `docs/DESIGN.md` for layout and rebuild commands).
- Private full-book research corpus available locally for evaluation lift studies.
- Sample comparison log (`data/eval/runs/sample_results_comparison.md`, append-only JSONL) updated when you run the sample or record-response scripts.
- Phase 2 (author-published essays in the public corpus) scaffolding shipped — manifest empty, awaiting curation.
- Demo surface for non-technical reviewers is the immediate next deliverable.

**Smoke checks (after `.env` is configured):**

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -c "from evaluation.connectivity import run_connectivity_check; print(run_connectivity_check()['ok'])"
```

For artefact paths, pipelines, and archived docs policy, see **[`docs/DESIGN.md`](docs/DESIGN.md)** (especially §2 layout, §5 evaluation, §6 private corpus). **Operator commands and where to edit scenarios** are in this file under **Quickstart** and **Scenarios** above.

---

## Disclaimer

This system is **decision support only**. It is **not** legal, financial, medical, or HR advice. Outputs must be reviewed by qualified human decision makers familiar with the local legal, regulatory, and organisational context. The system never automates a decision; it offers a structured, citable second perspective.

---

## Author

**Rommel Sharma** — Enterprise Technology Leader | Applied AI Practitioner

[LinkedIn](https://www.linkedin.com/in/rommelsharma/) · Open to collaboration on decision-intelligence and responsible-AI projects.
