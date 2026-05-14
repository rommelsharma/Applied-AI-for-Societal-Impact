"""Lightweight eval metrics for bias-detector JSON (Phase 1 eval spine)."""

from __future__ import annotations

import os
from typing import Any


def _taxonomy_bias_names(taxonomy: list[dict]) -> set[str]:
    names: set[str] = set()
    for row in taxonomy:
        key = row.get("bias_name")
        if isinstance(key, str) and key.strip():
            names.add(key.strip())
    return names


def _validate_bias_entry(entry: Any) -> list[str]:
    errs: list[str] = []
    if not isinstance(entry, dict):
        return ["bias entry is not an object"]
    required = ("bias_name", "bias_layer", "confidence", "explanation", "evidence", "risk")
    for field in required:
        if field not in entry:
            errs.append(f"missing field biases_identified[].{field}")
    if entry.get("bias_layer") not in ("human", "system", None):
        if "bias_layer" in entry:
            errs.append("bias_layer must be 'human' or 'system'")
    if entry.get("confidence") not in ("high", "medium", "low", None):
        if "confidence" in entry:
            errs.append("confidence must be high|medium|low")
    return errs


def validate_detector_payload(payload: Any, *, taxonomy_names: set[str]) -> dict[str, Any]:
    """Schema + taxonomy checks for one model payload (without_rag or with_rag)."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return {"schema_ok": False, "errors": ["payload is not a JSON object"], "bias_count": 0, "taxonomy_violations": []}

    required_top = (
        "situation_summary",
        "biases_identified",
        "noise_considerations",
        "overall_risk_assessment",
        "recommended_actions",
        "alternative_perspectives",
    )
    for key in required_top:
        if key not in payload:
            errors.append(f"missing top-level key: {key}")

    if "situation_summary" in payload and not isinstance(payload.get("situation_summary"), str):
        errors.append("situation_summary must be a string")
    if "overall_risk_assessment" in payload and not isinstance(payload.get("overall_risk_assessment"), str):
        errors.append("overall_risk_assessment must be a string")

    biases = payload.get("biases_identified")
    if not isinstance(biases, list):
        errors.append("biases_identified must be a list")
        biases = []

    taxonomy_violations: list[str] = []
    for i, entry in enumerate(biases):
        errors.extend([f"[{i}] {e}" for e in _validate_bias_entry(entry)])
        if isinstance(entry, dict):
            name = entry.get("bias_name")
            if isinstance(name, str) and name and name not in taxonomy_names:
                taxonomy_violations.append(name)

    noise = payload.get("noise_considerations")
    if not isinstance(noise, dict):
        errors.append("noise_considerations must be an object")
    else:
        if "present" not in noise:
            errors.append("noise_considerations.present missing")
        if "explanation" not in noise:
            errors.append("noise_considerations.explanation missing")

    for key in ("recommended_actions", "alternative_perspectives"):
        val = payload.get(key)
        if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
            errors.append(f"{key} must be a list of strings")

    return {
        "schema_ok": not errors and not taxonomy_violations,
        "errors": errors,
        "bias_count": len(biases),
        "taxonomy_violations": taxonomy_violations,
    }


def score_groundedness_lexical(with_rag: dict[str, Any], retrieval: list[dict[str, Any]]) -> dict[str, Any]:
    """Heuristic: share of identified biases whose canonical name appears in retrieved text blobs."""
    biases = with_rag.get("biases_identified") if isinstance(with_rag, dict) else None
    if not isinstance(biases, list) or not biases:
        return {"groundedness": None, "supported": 0, "total": 0, "pass": True}

    parts: list[str] = []
    for r in retrieval:
        if not isinstance(r, dict):
            continue
        parts.append(str(r.get("summary", "")))
        parts.append(str(r.get("text_excerpt", "")))
        parts.append(str(r.get("text", "")))
    blob = " ".join(parts).lower()

    supported = 0
    for entry in biases:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("bias_name", "")).replace("_", " ").strip().lower()
        if not name:
            continue
        if name in blob or name.replace(" ", "_") in blob:
            supported += 1

    total = len([b for b in biases if isinstance(b, dict) and b.get("bias_name")])
    g = supported / total if total else 1.0
    threshold = float(os.getenv("EVAL_GROUNDEDNESS_THRESHOLD", "0.35"))
    return {
        "groundedness": g,
        "supported": supported,
        "total": total,
        "threshold": threshold,
        "pass": g >= threshold if total else True,
    }


def score_comparison_result(
    result: dict[str, Any],
    *,
    taxonomy: list[dict],
) -> dict[str, Any]:
    """Score a full ``detect_bias_comparison`` payload."""
    names = _taxonomy_bias_names(taxonomy)
    no_rag = result.get("without_rag")
    with_rag = result.get("with_rag")
    retrieval = result.get("retrieval") or []

    m_no = validate_detector_payload(no_rag, taxonomy_names=names)
    m_yes = validate_detector_payload(with_rag, taxonomy_names=names)

    scores = [float(r.get("score", 0.0)) for r in retrieval if isinstance(r, dict)]
    avg_top_score = sum(scores) / len(scores) if scores else None
    hybrid_hits = sum(1 for r in retrieval if isinstance(r, dict) and r.get("retrieval_method") == "hybrid")

    grounded = score_groundedness_lexical(with_rag if isinstance(with_rag, dict) else {}, retrieval)

    return {
        "without_rag": m_no,
        "with_rag": m_yes,
        "retrieval": {
            "chunk_count": len(retrieval),
            "avg_score": avg_top_score,
            "hybrid_chunk_hits": hybrid_hits,
        },
        "groundedness": grounded,
        "query_classification": result.get("query_classification"),
        "aggregate_schema_ok": m_no["schema_ok"] and m_yes["schema_ok"],
    }


def aggregate_run_metrics(per_scenario: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up per-scenario metric dicts (output of :func:`score_comparison_result`)."""
    total = len(per_scenario)
    ok = sum(1 for row in per_scenario if row.get("aggregate_schema_ok"))
    bias_delta = []
    for row in per_scenario:
        m = row.get("with_rag") or {}
        n = row.get("without_rag") or {}
        if isinstance(m, dict) and isinstance(n, dict):
            bc_m = m.get("bias_count")
            bc_n = n.get("bias_count")
            if isinstance(bc_m, int) and isinstance(bc_n, int):
                bias_delta.append(bc_m - bc_n)
    grounded_ok = sum(1 for row in per_scenario if (row.get("groundedness") or {}).get("pass", True))
    return {
        "scenario_count": total,
        "all_schema_ok": ok == total and total > 0,
        "schema_ok_count": ok,
        "groundedness_pass_count": grounded_ok,
        "bias_count_delta_with_minus_without_rag": bias_delta,
    }
