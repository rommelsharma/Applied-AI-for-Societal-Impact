"""Streamlit UI — DecisionLens: Decision Intelligence Copilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# ── path bootstrap (works whether run via `streamlit run ui/app.py` or Docker) ─
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# ── page config — must be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="DecisionLens",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── project imports (after sys.path is set) ────────────────────────────────────
from app.services.bias_detector import detect_bias_comparison, load_taxonomy
from evaluation.metrics import score_comparison_result
from evaluation.run_history import load_recent_runs, load_run, save_ui_run
from evaluation.scenario_catalog import (
    get_baseline_scenarios,
    get_private_book_questions,
    scenario_text_for_detection,
    scenario_title,
)

# ── constants ─────────────────────────────────────────────────────────────────
_CONF_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}
_LAYER_LABEL = {"human": "Cognitive", "system": "Systemic"}
_DOMAIN_ICON = {
    "hiring": "👥",
    "ai_governance": "🤖",
    "strategy": "📊",
    "performance": "📈",
    "legal": "⚖️",
    "leadership": "🎯",
    "compliance": "📋",
}

_PLACEHOLDER = (
    "Describe a real decision situation in plain language.\n\n"
    "Example: A hiring panel is split on two finalists. Interviewers were asked "
    "to assess 'culture fit' with no scoring rubric. One finalist shares several "
    "interviewers' hobbies and alma mater; the other has stronger documented "
    "outcomes but is quieter in conversation. The panel wants to decide today."
)


# ── corpus freshness check (no Bedrock calls) ─────────────────────────────────

def _corpus_status(corpus: str) -> dict:
    """
    Fast file-only check against corpus_fingerprint.json.
    Returns a dict with keys: up_to_date, new_files, changed_files,
    removed_files, unparsed_pdfs, fingerprint_exists.
    Never raises — silently returns up_to_date=True on any error.
    """
    try:
        import hashlib
        from shared_components.utilities.path_utils import (
            get_corpus_dir,
            get_parsed_text_dir,
            get_raw_data_dir,
        )

        def _sha256(path: Path) -> str:
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for block in iter(lambda: fh.read(65_536), b""):
                    h.update(block)
            return h.hexdigest()

        parsed_dir = get_parsed_text_dir(corpus)
        raw_dir = get_raw_data_dir(corpus)
        fp_path = get_corpus_dir(corpus) / "corpus_fingerprint.json"

        if not fp_path.is_file():
            return {"up_to_date": False, "fingerprint_exists": False,
                    "new_files": [], "changed_files": [], "removed_files": [], "unparsed_pdfs": []}

        import json as _json
        known: dict = _json.loads(fp_path.read_text(encoding="utf-8")).get("sources", {})

        current_stems = {p.stem for p in parsed_dir.glob("*.txt")} if parsed_dir.is_dir() else set()
        new_files = sorted(current_stems - set(known))
        removed = sorted(set(known) - current_stems)

        # Changed = size differs (fast proxy before full hash)
        changed = []
        for stem in current_stems & set(known):
            txt = parsed_dir / known[stem]["file"]
            if txt.is_file() and txt.stat().st_size != known[stem].get("size_bytes", -1):
                changed.append(stem)

        parsed_stems = {p.stem for p in parsed_dir.glob("*.txt")} if parsed_dir.is_dir() else set()
        unparsed = (
            [p.stem for p in sorted(raw_dir.glob("*.pdf")) if p.stem not in parsed_stems]
            if raw_dir.is_dir() else []
        )

        ok = not new_files and not changed and not removed and not unparsed
        return {"up_to_date": ok, "fingerprint_exists": True,
                "new_files": new_files, "changed_files": changed,
                "removed_files": removed, "unparsed_pdfs": unparsed}
    except Exception:
        return {"up_to_date": True, "fingerprint_exists": True,
                "new_files": [], "changed_files": [], "removed_files": [], "unparsed_pdfs": []}


# ── cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _taxonomy() -> list[dict]:
    return load_taxonomy()


@st.cache_data(ttl=3600, show_spinner=False)
def _catalog() -> list[dict]:
    try:
        return get_baseline_scenarios() + get_private_book_questions()
    except Exception:
        return []


@st.cache_data(ttl=60, show_spinner=False)
def _run_history() -> list[dict]:
    """Load index records for the 20 most recent UI runs (no result blobs)."""
    try:
        return load_recent_runs(n=20)
    except Exception:
        return []


def _dropdown_label(entry: dict) -> str:
    """Human-readable dropdown label for any catalog entry."""
    base = scenario_title(entry)
    # For private-book entries, append the first ~70 chars of the question
    # so the user can distinguish entries from the same book.
    question = entry.get("question", "")
    if question and entry.get("source_book"):
        preview = question[:70].rstrip() + ("…" if len(question) > 70 else "")
        return f"{base}  —  {preview}"
    return base


# ── render helpers ─────────────────────────────────────────────────────────────

def _render_bias_card(entry: dict, *, show_provenance: bool = False) -> None:
    name = entry.get("bias_name", "unknown")
    conf = entry.get("confidence", "low")
    layer = entry.get("bias_layer", "human")
    icon = _CONF_ICON.get(conf, "⚪")
    layer_label = _LAYER_LABEL.get(layer, layer)

    # Grounded badge — shown only when the field is present (i.e. RAG mode)
    grounded = entry.get("grounded")
    if show_provenance and grounded is not None:
        grounded_badge = " &nbsp;·&nbsp; 📌 grounded" if grounded else " &nbsp;·&nbsp; 🔘 model knowledge"
    else:
        grounded_badge = ""

    header = f"{icon} **{name}** &nbsp;·&nbsp; {layer_label} &nbsp;·&nbsp; {conf} confidence{grounded_badge}"
    with st.expander(header, expanded=(conf == "high")):
        cols = st.columns([1, 1])
        with cols[0]:
            st.markdown("**Explanation**")
            st.markdown(entry.get("explanation") or "—")
            st.markdown("**Evidence**")
            st.markdown(entry.get("evidence") or "—")
        with cols[1]:
            st.markdown("**Risk if unaddressed**")
            st.markdown(entry.get("risk") or "—")

        # Supporting passage — shown when present and non-empty
        if show_provenance:
            passage = entry.get("supporting_passage") or ""
            chunks_used = entry.get("source_chunks") or []
            if passage.strip():
                st.markdown("**📖 Supporting passage from knowledge base**")
                st.info(f'"{passage}"')
                if chunks_used:
                    st.caption("Source chunks: " + ", ".join(f"`{c}`" for c in chunks_used))
            elif grounded is False:
                st.caption(
                    "🔘 No matching passage retrieved for this bias. "
                    "Finding is based on scenario analysis — verify against domain literature."
                )


def _render_provenance_banner(provenance: dict) -> None:
    """Render the retrieval provenance block above the RAG analysis column."""
    chunks_n = provenance.get("chunks_retrieved", 0)
    mean_score = provenance.get("mean_similarity_score")
    gap = provenance.get("retrieval_gap", False)
    gap_note = provenance.get("gap_note", "")
    gnd = provenance.get("groundedness_score")

    score_str = f"{mean_score:.3f}" if mean_score is not None else "—"
    gnd_str = f"{gnd:.0%}" if gnd is not None else "—"

    if gap and gap_note:
        st.warning(
            f"⚠️ **Retrieval gap** — {chunks_n} chunks retrieved · "
            f"avg similarity {score_str} · groundedness {gnd_str}\n\n{gap_note}",
            icon="🔘",
        )
    else:
        st.success(
            f"📚 {chunks_n} chunks retrieved · avg similarity {score_str} · "
            f"groundedness {gnd_str} · findings below are source-attributed",
            icon="📌",
        )


def _render_analysis(payload: dict, heading: str, *, show_provenance: bool = False) -> None:
    st.subheader(heading)
    if not isinstance(payload, dict):
        st.error("Invalid response payload.")
        return

    summary = payload.get("situation_summary")
    if summary:
        st.info(summary)

    biases: list[dict] = payload.get("biases_identified") or []
    high = [b for b in biases if b.get("confidence") == "high"]
    med = [b for b in biases if b.get("confidence") == "medium"]
    low = [b for b in biases if b.get("confidence") == "low"]

    if biases:
        st.markdown(
            f"**{len(biases)} bias{'es' if len(biases) != 1 else ''} identified** — "
            f"🔴 {len(high)} high · 🟠 {len(med)} medium · 🟡 {len(low)} low"
        )
        for b in biases:
            _render_bias_card(b, show_provenance=show_provenance)
    else:
        st.caption("No biases identified.")

    noise = payload.get("noise_considerations")
    if isinstance(noise, dict) and noise.get("present"):
        with st.expander("🔊 Noise considerations"):
            st.markdown(noise.get("explanation") or "—")

    risk = payload.get("overall_risk_assessment")
    if risk:
        st.markdown("**Overall risk assessment**")
        st.warning(risk)

    actions: list = payload.get("recommended_actions") or []
    if actions:
        st.markdown("**Recommended actions**")
        for a in actions:
            st.markdown(f"- {a}")

    alts: list = payload.get("alternative_perspectives") or []
    if alts:
        with st.expander("🔄 Alternative perspectives to consider"):
            for a in alts:
                st.markdown(f"- {a}")


def _render_retrieval(chunks: list[dict]) -> None:
    if not chunks:
        st.caption("No chunks retrieved.")
        return
    for i, chunk in enumerate(chunks, 1):
        method = chunk.get("retrieval_method", "dense")
        badge = "🔵 hybrid" if method == "hybrid" else "⚪ dense"
        source = chunk.get("source") or "unknown"
        author = chunk.get("author") or ""
        score = chunk.get("score") or 0.0
        concepts = ", ".join((chunk.get("concepts") or [])[:5]) or "—"
        biases_tagged = ", ".join((chunk.get("biases") or [])[:4]) or "—"
        header = f"{badge} · **{source}** ({author}) · score {score:.3f}"
        with st.expander(f"[{i}] {header}"):
            meta_cols = st.columns(2)
            with meta_cols[0]:
                st.caption(f"**Concepts:** {concepts}")
                st.caption(f"**Biases tagged:** {biases_tagged}")
            with meta_cols[1]:
                st.caption(f"**Passage type:** {chunk.get('passage_type') or '—'}")
                st.caption(f"**Decision phase:** {chunk.get('decision_phase') or '—'}")
            summary = chunk.get("summary")
            if summary:
                st.markdown(f"**Summary:** {summary}")
            excerpt = chunk.get("text_excerpt") or ""
            if excerpt:
                st.markdown("**Excerpt**")
                st.code(excerpt[:700], language=None)


def _render_metrics(result: dict) -> None:
    taxonomy = _taxonomy()
    m = score_comparison_result(result, taxonomy=taxonomy)
    no_r = m.get("without_rag") or {}
    wi_r = m.get("with_rag") or {}
    ret_m = m.get("retrieval") or {}
    gnd = m.get("groundedness") or {}
    qc = result.get("query_classification") or {}

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Biases — baseline", no_r.get("bias_count", 0))
    delta = (wi_r.get("bias_count") or 0) - (no_r.get("bias_count") or 0)
    c2.metric("Biases — with RAG", wi_r.get("bias_count", 0), delta=delta)
    c3.metric("Chunks retrieved", ret_m.get("chunk_count", 0))
    hybrid_hits = ret_m.get("hybrid_chunk_hits", 0)
    c4.metric("Hybrid hits", hybrid_hits)
    g = gnd.get("groundedness")
    c5.metric("Groundedness", f"{g:.0%}" if g is not None else "—")
    schema_ok = m.get("aggregate_schema_ok", False)
    c6.metric("Schema valid", "✅" if schema_ok else "❌")

    if qc.get("intent"):
        st.caption(f"Query intent: **{qc['intent']}** · MMR λ used: **{qc.get('mmr_lambda', '—')}**")


# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://img.shields.io/badge/Bias--Aware%20RAG-v4.0-blue?style=flat-square",
        use_container_width=False,
    )
    st.markdown("### ⚙️ Settings")

    corpus = st.selectbox(
        "Knowledge corpus",
        options=["public", "private"],
        index=0,
        help=(
            "**public** — synthesised research dossiers (committed to repo, always available).\n\n"
            "**private** — full-book PDFs (local only, gitignored). "
            "Must run the ingestion pipeline first."
        ),
    )

    top_k = st.slider(
        "Chunks to retrieve (top-k)",
        min_value=2,
        max_value=20,
        value=8,
        step=1,
        help="How many knowledge-base chunks to inject into the RAG context.",
    )

    with st.expander("Advanced retrieval"):
        mmr_lambda = st.slider(
            "MMR λ — relevance ↔ diversity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="1.0 = pure relevance. 0.5 = balanced. 0.0 = pure diversity.",
        )
        st.caption(
            "BM25, reranker, overlap filter and other toggles are read from `.env`."
        )

    st.divider()
    st.caption(
        "**Disclaimer:** Decision support only. Not legal, HR, medical, or financial advice. "
        "Always pair outputs with qualified human judgment and domain expertise."
    )
    st.caption("Built by Rommel Sharma · AWS Bedrock + FAISS + Claude")

    # ── corpus freshness indicator ────────────────────────────────────────────
    st.divider()
    st.markdown("**Corpus status**")
    status = _corpus_status(corpus)

    if not status["fingerprint_exists"]:
        st.warning(
            "No fingerprint found. Run once to initialise:\n\n"
            f"`python scripts/check_corpus_updates.py --corpus {corpus}`",
            icon="🗂️",
        )
    elif status["up_to_date"]:
        st.success("Index is up to date.", icon="✅")
    else:
        parts = []
        if status["new_files"]:
            parts.append(f"**{len(status['new_files'])} new** file(s): "
                         + ", ".join(f"`{f}`" for f in status["new_files"][:3])
                         + ("…" if len(status["new_files"]) > 3 else ""))
        if status["changed_files"]:
            parts.append(f"**{len(status['changed_files'])} changed** file(s): "
                         + ", ".join(f"`{f}`" for f in status["changed_files"][:3]))
        if status["removed_files"]:
            parts.append(f"**{len(status['removed_files'])} removed** from last index")
        if status["unparsed_pdfs"]:
            parts.append(f"**{len(status['unparsed_pdfs'])} PDF(s)** not yet parsed")
        msg = "\n\n".join(parts) + (
            f"\n\nRebuild with:\n\n"
            f"`python scripts/check_corpus_updates.py --corpus {corpus} --ingest`"
        )
        st.warning(msg, icon="⚠️")


# ── main ──────────────────────────────────────────────────────────────────────

st.title("⚖️ DecisionLens")
st.markdown(
    "Evidence-grounded decision intelligence — surfaces the cognitive and structural "
    "patterns most likely to distort judgment, with full source attribution."
)
st.divider()

# ── scenario input ─────────────────────────────────────────────────────────────

# Flush any pending catalog load BEFORE the text_area widget is instantiated.
# Streamlit forbids writing to a widget key after it has been rendered; the
# staging key (_pending_scenario) sidesteps that by carrying the value across
# the rerun boundary.
if "_pending_scenario" in st.session_state:
    st.session_state["scenario_input"] = st.session_state.pop("_pending_scenario")
elif "scenario_input" not in st.session_state:
    st.session_state["scenario_input"] = ""

input_col, catalog_col = st.columns([3, 2], gap="large")

with input_col:
    st.markdown("#### Describe the decision")
    scenario_text: str = st.text_area(
        label="Scenario",
        placeholder=_PLACEHOLDER,
        height=175,
        label_visibility="collapsed",
        key="scenario_input",
    )

with catalog_col:
    st.markdown("#### Or load an example")
    catalog = _catalog()
    if catalog:
        options: dict[str, dict] = {_dropdown_label(s): s for s in catalog}
        selected_label = st.selectbox(
            "Example scenarios",
            options=["— choose a scenario —"] + list(options.keys()),
            label_visibility="collapsed",
        )
        if selected_label != "— choose a scenario —":
            entry = options[selected_label]
            domain = entry.get("domain", "")
            icon = _DOMAIN_ICON.get(domain, "📋")
            tags = entry.get("tags") or []
            st.caption(f"{icon} **{domain}**" + (f"  ·  tags: {', '.join(tags[:4])}" if tags else ""))
            if st.button("↙ Load into editor", use_container_width=True):
                st.session_state["_pending_scenario"] = scenario_text_for_detection(entry)
                st.rerun()
    else:
        st.info(
            "Scenario catalog not found. "
            "Run `python scripts/build_scenarios_catalog.py` from the project root."
        )

# ── run button ────────────────────────────────────────────────────────────────

st.markdown("")
run_col, _ = st.columns([1, 5])
with run_col:
    run_clicked = st.button("▶ Analyse", type="primary", use_container_width=True)

# ── execute ────────────────────────────────────────────────────────────────────

if run_clicked:
    text = (st.session_state.get("scenario_input") or "").strip()
    if not text:
        st.warning("Please enter or load a scenario first.")
        st.stop()

    with st.spinner("Running baseline + RAG analysis via Amazon Bedrock … this takes ~15–30 s"):
        try:
            result = detect_bias_comparison(
                text,
                corpus=corpus,
                top_k=top_k,
                mmr_lambda=mmr_lambda,
            )
            st.session_state["last_result"] = result
            st.session_state["last_scenario"] = text

            # Persist run to data/eval/runs/ and refresh the history cache
            try:
                saved_path = save_ui_run(
                    scenario=text,
                    corpus=corpus,
                    top_k=top_k,
                    mmr_lambda=mmr_lambda,
                    result=result,
                )
                st.session_state["last_saved_path"] = str(saved_path)
            except Exception:
                # Never let a save failure block the user from seeing results
                st.session_state.pop("last_saved_path", None)

            # Bust the run history cache so the Recent Runs panel refreshes
            st.cache_data.clear()

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

# ── results ────────────────────────────────────────────────────────────────────

if "last_result" in st.session_state:
    result: dict = st.session_state["last_result"]
    displayed_scenario: str = st.session_state.get("last_scenario", "")

    st.divider()

    # Metrics bar
    _render_metrics(result)

    st.divider()

    # Side-by-side analysis
    provenance = result.get("retrieval_provenance") or {}
    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        _render_analysis(result.get("without_rag") or {}, "Without RAG — Baseline")
    with right_col:
        if provenance:
            _render_provenance_banner(provenance)
        _render_analysis(
            result.get("with_rag") or {},
            "With RAG — Knowledge-Enhanced",
            show_provenance=True,
        )

    st.divider()

    # Retrieved chunks
    chunks = result.get("retrieval") or []
    with st.expander(f"📚 Retrieved knowledge-base chunks ({len(chunks)})", expanded=False):
        _render_retrieval(chunks)

    # Download + save confirmation
    with st.expander("⬇️ Export full JSON result"):
        saved_path = st.session_state.get("last_saved_path")
        if saved_path:
            st.caption(f"💾 Auto-saved to `{Path(saved_path).name}`")
        st.download_button(
            label="Download result.json",
            data=json.dumps(result, indent=2),
            file_name="bias_analysis_result.json",
            mime="application/json",
            use_container_width=True,
        )

# ── recent runs panel ─────────────────────────────────────────────────────────

st.divider()
history = _run_history()
run_header = f"🕑 Recent runs ({len(history)} of last 20)"

with st.expander(run_header, expanded=False):
    if not history:
        st.caption(
            "No saved runs yet. Run your first scenario above — "
            "results are automatically saved to `data/eval/runs/`."
        )
    else:
        # Column header row
        hdr = st.columns([2, 1, 1, 1, 1, 1, 1])
        hdr[0].caption("**Scenario**")
        hdr[1].caption("**Corpus**")
        hdr[2].caption("**Baseline**")
        hdr[3].caption("**RAG**")
        hdr[4].caption("**Lift**")
        hdr[5].caption("**Grounded**")
        hdr[6].caption("**Reload**")

        for rec in history:
            # Parse saved_at to a friendly local string
            try:
                from datetime import datetime as _dt
                ts_raw = rec.get("saved_at", "")
                ts_disp = _dt.fromisoformat(ts_raw).strftime("%d %b %H:%M") if ts_raw else "—"
            except Exception:
                ts_disp = rec.get("saved_at", "—")[:16]

            preview = rec.get("scenario_preview") or "—"
            corpus_tag = rec.get("corpus") or "—"
            bc_base = rec.get("bias_count_baseline")
            bc_rag = rec.get("bias_count_rag")
            lift = rec.get("bias_lift")
            gnd = rec.get("groundedness_score")
            gap = rec.get("retrieval_gap", False)

            gnd_str = (
                f"{'⚠️ ' if gap else ''}{gnd:.0%}" if gnd is not None else "—"
            )
            lift_str = f"+{lift}" if (lift is not None and lift > 0) else str(lift) if lift is not None else "—"

            cols = st.columns([2, 1, 1, 1, 1, 1, 1])
            cols[0].markdown(
                f"<small><b>{ts_disp}</b> &nbsp; {preview[:65]}{'…' if len(preview) > 65 else ''}</small>",
                unsafe_allow_html=True,
            )
            cols[1].caption(corpus_tag)
            cols[2].caption(str(bc_base) if bc_base is not None else "—")
            cols[3].caption(str(bc_rag) if bc_rag is not None else "—")
            cols[4].caption(lift_str)
            cols[5].caption(gnd_str)

            run_key = f"reload_{rec['file']}"
            if cols[6].button("↩ Load", key=run_key, use_container_width=True):
                try:
                    full = load_run(rec["path"])
                    st.session_state["last_result"] = full["result"]
                    st.session_state["last_scenario"] = full.get("scenario", "")
                    st.session_state["last_saved_path"] = rec["path"]
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not reload run: {exc}")
