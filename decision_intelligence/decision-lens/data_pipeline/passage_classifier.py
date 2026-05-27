"""
Lightweight passage classification used during chunk enrichment.

Two orthogonal signals are produced for every chunk:

    * ``passage_type``   - one of {prescription, framework, study, case_study,
                           anecdote, definition}.
    * ``decision_phase`` - one of {diagnose, design, decide, review}.

Why this matters:
    * Retrieval can prefer prescriptive content over narrative when the user
      asks "what should I do?", and prefer framework / study content when the
      user asks "what is happening here?".
    * Downstream UI surfaces (e.g. tabs for "Diagnostics", "Recommended
      Actions", "Worked Studies") can pivot off these tags.

Two implementations are provided:

    * :func:`classify_rule_based` - free, deterministic, runs at indexing time
      with zero network calls. Tuned for the dossier vocabulary and the kind
      of decision-science prose found in the curated corpus.
    * :func:`classify_with_llm` - calls Claude Haiku via Bedrock when the
      operator opts in via ``ENRICHMENT_USE_LLM=true``. Higher quality, costs
      a few cents per thousand chunks.

The dispatcher :func:`classify_passage` picks whichever path is configured.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from shared_components.settings import RAG_SETTINGS


PASSAGE_TYPES = ("prescription", "framework", "study", "case_study", "anecdote", "definition")
DECISION_PHASES = ("diagnose", "design", "decide", "review")


# Heuristic vocabularies. Order matters: more specific patterns first so that
# e.g. a prescriptive sentence inside a study chunk is still tagged as a
# study rather than a prescription.
_PASSAGE_RULES: list[tuple[str, re.Pattern]] = [
    (
        "study",
        re.compile(
            r"\b(study|studies|experiment|experiments?|researchers?|"
            r"meta[- ]analysis|sample\s+of|participants|subjects|trial|finding[s]?)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "case_study",
        re.compile(
            r"\b(case\s+(of|study)|the\s+\w+\s+case|in\s+re\s+\w+|v\.\s+\w+|"
            r"plaintiff|defendant|on\s+\w+\s+\d{1,2},\s+\d{4})\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "framework",
        re.compile(
            r"\b(model|framework|principle[s]?|step\s+\d+|stages?|phases?|"
            r"taxonomy|categories|dimensions|pillars|the\s+\w+\s+model)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "prescription",
        re.compile(
            r"\b(should|must|ought\s+to|recommend|recommendations?|"
            r"best\s+practice|to\s+(?:reduce|avoid|mitigate|improve|prevent)|"
            r"how\s+to|practice|guideline|checklist|rubric|protocol)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "definition",
        re.compile(
            r"\b(is\s+defined\s+as|refers\s+to|means\s+that|"
            r"in\s+other\s+words|that\s+is,|namely|known\s+as)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "anecdote",
        re.compile(
            r"\b(once|one\s+day|years\s+ago|i\s+recall|she\s+told|he\s+said|"
            r"my\s+\w+|let\s+me\s+share|imagine\s+(?:you|a)\b)",
            flags=re.IGNORECASE,
        ),
    ),
]

_DECISION_PHASE_RULES: list[tuple[str, re.Pattern]] = [
    (
        "diagnose",
        re.compile(
            r"\b(identify|recognize|spot|warning\s+sign|symptom|pattern|"
            r"detect|diagnos[ei]|red\s+flag|cue[s]?\b|signal[s]?\b)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "design",
        re.compile(
            r"\b(design|build|structure|architecture|set\s+up|establish|"
            r"create\s+a\s+process|implement|institut[e|ion])\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "decide",
        re.compile(
            r"\b(decide|decision|choose|select|weigh|trade[- ]off|"
            r"verdict|sentence|judgment\s+call|conclude)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "review",
        re.compile(
            r"\b(review|audit|post[- ]mortem|after[- ]action|retrospective|"
            r"evaluat(?:e|ion)|reassess|check\s+back|lessons\s+learned|"
            r"feedback\s+loop)\b",
            flags=re.IGNORECASE,
        ),
    ),
]


def classify_rule_based(text: str) -> dict[str, str | float]:
    """Return ``passage_type`` and ``decision_phase`` using regex heuristics.

    Each rule scores its category by counting non-overlapping matches in the
    text. The highest-scoring category wins; if no rule matches, the
    category falls back to ``"unknown"`` and confidence is reported as 0.

    The result is intentionally cheap to compute - this runs once per chunk
    at index-build time. Confidence is a coarse 0-1 signal usable for
    downstream filters.
    """
    passage_scores = {category: len(pattern.findall(text)) for category, pattern in _PASSAGE_RULES}
    phase_scores = {category: len(pattern.findall(text)) for category, pattern in _DECISION_PHASE_RULES}

    passage_type, passage_hits = max(passage_scores.items(), key=lambda item: item[1])
    decision_phase, phase_hits = max(phase_scores.items(), key=lambda item: item[1])

    if passage_hits == 0:
        passage_type = "unknown"
    if phase_hits == 0:
        decision_phase = "unknown"

    # Normalise hits to a 0-1 confidence by dividing by an empirical ceiling.
    # 5 hits is "very confident"; anything above saturates.
    passage_confidence = min(passage_hits / 5.0, 1.0)
    phase_confidence = min(phase_hits / 5.0, 1.0)

    return {
        "passage_type": passage_type,
        "passage_confidence": round(passage_confidence, 2),
        "decision_phase": decision_phase,
        "decision_phase_confidence": round(phase_confidence, 2),
        "classifier": "rule",
    }


@lru_cache(maxsize=1)
def _bedrock_provider():
    """Lazy, cached Bedrock client for the LLM classifier path.

    Imported lazily so the heuristic path stays free of any AWS dependency
    when ``ENRICHMENT_USE_LLM`` is off.
    """
    from app.services.bedrock_provider import BedrockProvider

    return BedrockProvider()


_LLM_CLASSIFIER_PROMPT = (
    "You are a document classifier. Given a chunk of decision-science writing, "
    "tag it with two labels chosen ONLY from the closed sets below. Return strict JSON.\n\n"
    "passage_type ∈ {prescription, framework, study, case_study, anecdote, definition, unknown}\n"
    "decision_phase ∈ {diagnose, design, decide, review, unknown}\n\n"
    "Definitions:\n"
    "- prescription: tells the reader what to do (\"should\", \"must\", recommended practices).\n"
    "- framework: presents a model, principle set, or named structure.\n"
    "- study: reports research findings, experiments, sample sizes, statistics.\n"
    "- case_study: narrates a real-world named case or legal proceeding.\n"
    "- anecdote: personal or illustrative story, no research grounding.\n"
    "- definition: defines or explains a term.\n\n"
    "- diagnose: helps the reader recognise a problem/bias/situation.\n"
    "- design: describes how to build a process, system, or structure.\n"
    "- decide: addresses the moment of choosing between options.\n"
    "- review: addresses post-decision evaluation, audit, or learning.\n\n"
    'Output schema (no markdown): {"passage_type": "...", "decision_phase": "...", '
    '"passage_confidence": 0.0-1.0, "decision_phase_confidence": 0.0-1.0}'
)


def classify_with_llm(text: str) -> dict[str, str | float]:
    """Use Claude Haiku to classify a chunk's passage type and decision phase.

    Falls back to the rule-based result if the LLM response is malformed or
    Bedrock is unavailable, so an LLM hiccup never aborts an index build.
    """
    provider = _bedrock_provider()

    try:
        result = provider.converse(
            system_prompt=_LLM_CLASSIFIER_PROMPT,
            user_prompt=f"Chunk text:\n{text[:4000]}\n\nReturn JSON only.",
            max_tokens=180,
        )
        payload = json.loads(result.text)
        passage_type = payload.get("passage_type", "unknown")
        decision_phase = payload.get("decision_phase", "unknown")
        if passage_type not in PASSAGE_TYPES + ("unknown",):
            passage_type = "unknown"
        if decision_phase not in DECISION_PHASES + ("unknown",):
            decision_phase = "unknown"
        return {
            "passage_type": passage_type,
            "passage_confidence": float(payload.get("passage_confidence", 0.0)),
            "decision_phase": decision_phase,
            "decision_phase_confidence": float(payload.get("decision_phase_confidence", 0.0)),
            "classifier": "llm",
        }
    except Exception as exc:  # pragma: no cover - LLM/transient failure
        # Heuristic fallback so a single chunk never breaks a 5,000-chunk
        # ingestion. The chunk still gets sensible tags, just lower quality.
        fallback = classify_rule_based(text)
        fallback["classifier_error"] = str(exc)
        return fallback


def classify_passage(text: str) -> dict[str, str | float]:
    """Pick the configured classifier (rule-based by default, LLM if opted-in)."""
    if RAG_SETTINGS.enrichment_use_llm:
        return classify_with_llm(text)
    return classify_rule_based(text)
