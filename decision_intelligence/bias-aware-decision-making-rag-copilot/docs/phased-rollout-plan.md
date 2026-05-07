# Phased Rollout Plan

This document operationalises **Section 6** of the earlier "Books as a Knowledge Source" analysis: a pragmatic, low-risk path to grow the public knowledge base while respecting copyright and keeping every release measurable. It is the canonical roadmap for corpus growth.

> The principle: the public corpus must always be safely shareable; the private corpus exists for evaluation lift; everything we publish in either should be measurable against the previous baseline.

---

## 1. Where We Are Today

| Artefact | Status | Where it lives |
|---|---|---|
| Synthesised dossiers (~10 books) | Built, indexed, public | `data/corpora/public/raw/*.pdf` |
| Full books (6) | Built, indexed, private only | `data/corpora/private/raw/*.pdf` |
| Public vector index | 301 chunks, IndexFlatIP, 1024 dims | `data/corpora/public/vector_store/` |
| Private vector index | 1,161 chunks, IndexFlatIP, 1024 dims | `data/corpora/private/vector_store/` |
| Public sample evaluation | Refreshed for the new corpus | `docs/sample_results_comparison.md` |
| Public-vs-private A/B harness | Implemented | `scripts/compare_versions.py` |

The current state is **Phase 0 + Phase 1** of the plan below. The next concrete delivery is **Phase 2**.

---

## 2. The Five Phases

### Phase 0 — Synthesised dossiers in public ✅ done

Short syntheses of seminal books, written from public sources, committed to `data/corpora/public/raw/` as PDFs. Provides domain coverage and the demo experience without copyright risk.

**Validation:** `python scripts/run_sample_comparison.py` produces a refreshable comparison report.

### Phase 1 — Full books in private corpus ✅ done

Full-text books ingested under `data/corpora/private/raw/`, never committed, used for measuring the lift over Phase 0 via `scripts/compare_versions.py`. This is the **research lab** — the place we learn what depth of context actually changes the model's reasoning.

**Validation:** `python scripts/compare_versions.py --versions public,private` quantifies the delta.

### Phase 2 — Author-published content in public corpus 🔜 next

Add lightweight, **license-clear** author-published content to the public corpus:

- Author essays on personal blogs.
- Transcripts of public lectures, podcasts, and panels.
- Preprints, working papers, and peer-reviewed articles where licensing permits.
- Open-access course material from universities.
- Self-published or Creative-Commons-licensed material.

The aim is to bridge the gap between Phase 0 (concept-level) and Phase 1 (book-level depth) without distributing copyrighted books. Each item is checked for license fit before inclusion. The scaffolding for this phase ships in this commit:

- `data/metadata/author_content_manifest.json` — the curated list, schema explained in §4.
- `scripts/ingest_author_content.py` — reads the manifest, fetches/normalises each item, writes `.txt` + `.meta.json` directly into `data/corpora/public/parsed/`, then the standard build pipeline takes over.

**Validation:** rebuild the public corpus, re-run `scripts/run_sample_comparison.py`, compare to the prior baseline; bias-count and recommended-action distributions should remain stable or improve.

### Phase 3 — Permitted summaries and reviews 🟡 future

Where licence cannot be obtained for author content but a clear public summary, book review, or commentary exists from a credible source (e.g. *The New York Review of Books*, *Aeon*, *Edge.org*), include it under the same manifest mechanism with `content_type: "summary"`. These are typically shorter than essays and benefit greatly from chapter-level chunking.

**Validation:** as Phase 2.

### Phase 4 — Scaling the index 🟡 future

When the public corpus passes ~3,000 chunks:

- Enable the LLM-as-judge reranker (`RAG_RERANKER=llm`) since the candidate pool will be large enough to benefit.
- Tune `RAG_RERANKER_CANDIDATES` and `RAG_TOP_K` against the evaluation harness.

When the public corpus passes ~50,000 chunks:

- Migrate the FAISS index from `IndexFlatIP` to `IndexIVFFlat` or `IndexHNSWFlat`.
- Re-run the comparison harness to confirm recall is preserved at the new precision settings.

---

## 3. Acceptance Criteria for Each Phase

A phase is "done" when:

1. The new artefacts are checked in (manifests, code, docs) — never the copyrighted source files.
2. The public vector index is rebuilt and committed under `data/corpora/public/vector_store/`.
3. `python scripts/run_sample_comparison.py` produces a refreshed `docs/sample_results_comparison.md` and the bias-count delta is non-negative.
4. `python scripts/compare_versions.py --versions <prev>,<current>` produces a comparison report stored under `data/corpora/comparisons/` for traceability.
5. The README's "Implemented now" section references the phase number that has just landed.

---

## 4. The Author-Content Manifest Schema

`data/metadata/author_content_manifest.json` is the single source of truth for what gets pulled into the public corpus from the open web. Schema:

```jsonc
{
  "version": "1.0",
  "license_policy": "Only items whose licence clearly permits research/portfolio use ...",
  "items": [
    {
      "slug": "unique-kebab-case-slug",         // becomes the .txt / .meta.json filename stem
      "title": "Article or lecture title",
      "author": "Author Name",
      "publication_year": "2018",
      "publisher": "Aeon | Edge.org | <author>.com",
      "content_type": "essay | lecture_transcript | preprint | summary | review",
      "source_url": "https://...",              // present when fetched from the web
      "local_path": "data/sourcing/<slug>.html", // present when already downloaded
      "license_status": "public_domain | creative_commons | author_self_published | permission_confirmed | review_required",
      "license_notes": "Free-text justification of license_status",
      "tags": ["anchoring", "premortem", "noise"], // optional - used to seed concept extraction
      "decision_domains": ["hr", "leadership"]    // optional - influences retrieval-domain filters
    }
  ]
}
```

Every item must have either `source_url` or `local_path`. No item with `license_status: review_required` is ingested — the script exits with an error referencing the item's slug.

---

## 5. The Author-Content Ingestion Script

`scripts/ingest_author_content.py` provides the Phase 2 entry point. It performs the same transform that `pdf_parser.py` does for PDFs — text extraction plus a `.meta.json` sidecar — but for HTML, plain text, or markdown content sourced from the open web.

Default invocation:

```bash
python scripts/ingest_author_content.py --corpus public --manifest data/metadata/author_content_manifest.json
```

Behaviour:

- Reads each item from the manifest.
- Skips any item with `license_status: review_required` (with a helpful error).
- Fetches `source_url` (HTML) or reads `local_path` (HTML/MD/TXT).
- Strips HTML to text (removes nav, header, footer, scripts, styles).
- Writes `data/corpora/<corpus>/parsed/<slug>.txt`.
- Writes a sibling `.meta.json` with the manifest fields plus `parser_notes: ["author_content_ingest"]`.
- Optional flags:
  - `--dry-run` — prints what it would do without writing anything.
  - `--only <slug>` — limit to a single item for incremental testing.
  - `--user-agent` — override the default polite UA string.
  - `--rate-limit 1.5` — seconds between fetches (default 1.0).

After the script completes, the standard pipeline continues:

```bash
# Skip parse step (PDFs already parsed); chunk + enrich + index pulls in the new .txt files.
python data_pipeline/build_knowledge_base.py --corpus public --skip-parse
python data_pipeline/build_vector_index.py --corpus public
```

A `--skip-parse` flag is added to `build_knowledge_base.py` so author-content ingest doesn't re-trigger the (slower) PDF parser.

---

## 6. License Discipline

The public corpus is part of a portfolio repository. The licensing rules are deliberately conservative:

- **Always include**: public-domain works, Creative Commons-licensed content, author self-published content (their own site / blog / Substack archive page), preprints with permissive licenses (arXiv, SSRN with author posting), and open-access journal articles.
- **Never include**: full books (even when out of print, even when "easy to find online").
- **Include after written confirmation**: copyrighted essays/articles with explicit author or publisher permission.
- **Investigate**: anything ambiguous — flagged in the manifest as `license_status: review_required`.

When in doubt, that item lives in the **private** corpus or not at all.

---

## 7. Sample Curation Targets (Not Yet Curated)

These are illustrative starting points the operator can review and add to the manifest with `license_status: review_required` until each is checked:

| Author | Likely sources | License likelihood |
|---|---|---|
| Daniel Kahneman | Edge.org essays, Nobel lecture transcript | Mostly permitted (Edge.org, Nobel public domain) |
| Olivier Sibony | His own site `oliviersibony.com`, McKinsey Quarterly co-authored pieces | Author site likely permitted; McKinsey requires confirmation |
| Cass Sunstein | SSRN preprints under his author profile | Generally permitted (author-posted) |
| Philip Tetlock | Good Judgment Project public reports | Mixed; check per-item |
| Robert Cialdini | His own site `influenceatwork.com`, public talks transcripts | Mixed; check per-item |
| Cathy O'Neil | `mathbabe.org` blog | Author-controlled |
| Bryan Stevenson | Equal Justice Initiative reports (eji.org) | Permitted with attribution |
| Caroline Criado Perez | Author's site, public columns | Mixed |
| Gary Klein | His own articles, white papers from `psychologyofintuition.com` | Mixed |
| Stuart Russell | His Berkeley/CHAI page, AAAI lectures (often CC) | Permitted |
| Yoshua Bengio | His own essays, Mila publications | Mostly permitted |
| Carlota Perez | Her own site `carlotaperez.org` | Author-controlled |

These are **not** yet ingested. The operator reviews each, confirms licence, then adds the item to the manifest with `license_status` set appropriately.

---

## 8. Measurement Plan

Every Phase 2 cohort (each batch of new author content) is evaluated against the **immediately preceding** corpus state, not against the original Phase 0:

```bash
# Tag the current public corpus before adding the cohort.
git tag corpus-before-cohort-N
python scripts/compare_versions.py --versions public,public_pre_cohort_N
```

The metrics that matter:

- **Bias-count delta** per scenario — should not regress.
- **Bias-name diversity** — broader is generally better, with a cap (more than ~12 biases per scenario suggests over-fitting).
- **Recommended-action count** and **Alternative-perspective count** — should remain ~ stable.
- **Schema fidelity** — must remain at 100%.
- **Average retrieved similarity score** — should rise modestly as the corpus densifies; sudden drops indicate concept-tag drift.

The comparison report goes into `data/corpora/comparisons/` and is reviewed before the new vector store is committed.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| License ambiguity creeps into public corpus | Manifest forces explicit `license_status`; script refuses to ingest `review_required` |
| Author content style differs from books and degrades retrieval | Profile-aware chunking already handles short essays via `dossier` profile |
| Adding noisy content reduces precision | Concept extraction is content-agnostic; the comparison harness catches regressions |
| Phase 2 outputs swing model behaviour unpredictably | Evaluation gating (§3) blocks merges that regress the canonical scenarios |
| Rate-limiting / scraping concerns | Script honours rate limits, identifies itself with a UA, and respects robots.txt for the source domain |

---

## 10. What Lands With This Document

In addition to this plan, the supporting code and data shipping in the same change:

- `data/metadata/author_content_manifest.json` (empty starter manifest with the schema).
- `scripts/ingest_author_content.py` (the fetch / normalise tool described above).
- `data_pipeline/build_knowledge_base.py` gains a `--skip-parse` flag so author-content ingest does not retrigger the PDF parser.
- The README's roadmap section is updated to mark Phase 0 and Phase 1 as done and Phase 2 as the next step.

When the operator chooses to start Phase 2, the workflow is:

1. Add one or more vetted entries to `author_content_manifest.json`.
2. Run `python scripts/ingest_author_content.py --corpus public`.
3. Run `python data_pipeline/build_knowledge_base.py --corpus public --skip-parse`.
4. Run `python data_pipeline/build_vector_index.py --corpus public`.
5. Run `python scripts/run_sample_comparison.py` to refresh the public sample report.
6. Run `python scripts/compare_versions.py --versions public_pre,public` and review the lift before committing the new vector store.

This is the operating loop for the rest of the corpus's growth.
