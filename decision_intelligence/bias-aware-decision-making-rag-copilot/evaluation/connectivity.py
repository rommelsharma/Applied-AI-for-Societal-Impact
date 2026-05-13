"""Pre-flight checks before expensive evaluation runs (Bedrock + vector store + retrieval)."""

from __future__ import annotations

import time
from typing import Any

from app.services.bedrock_provider import (
    BedrockConfigurationError,
    BedrockInferenceError,
    BedrockProvider,
)
from rag.retriever import KnowledgeRetriever
from shared_components.settings import BEDROCK_SETTINGS, RAG_SETTINGS


def run_connectivity_check(*, retrieval_probe_query: str = "structured interview rubric hiring bias") -> dict[str, Any]:
    """Return a structured report suitable for persisting next to response captures.

    Steps:
        1. Bedrock credentials present (no network yet).
        2. Embedding API round-trip (Titan v2).
        3. Vector store load for the active corpus.
        4. Single retrieval query to prove index + metadata alignment.
    """
    steps: list[dict[str, Any]] = []
    errors: list[str] = []

    # Step 1 — configuration
    if not BEDROCK_SETTINGS.has_api_key:
        errors.append("AWS_BEARER_TOKEN_BEDROCK is not set.")
        steps.append({"name": "bedrock_credentials", "ok": False})
        return {"ok": False, "steps": steps, "errors": errors}

    steps.append({"name": "bedrock_credentials", "ok": True, "region": BEDROCK_SETTINGS.region})

    # Step 2 — embedding
    try:
        provider = BedrockProvider()
        t0 = time.perf_counter()
        vec = provider.embed_text("connectivity probe: decision quality and bias awareness")
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        dim = len(vec)
        if dim != BEDROCK_SETTINGS.embedding_dimensions:
            errors.append(f"Embedding dimension mismatch: got {dim}, expected {BEDROCK_SETTINGS.embedding_dimensions}.")
        steps.append(
            {
                "name": "embedding_probe",
                "ok": dim == BEDROCK_SETTINGS.embedding_dimensions,
                "latency_ms": elapsed_ms,
                "dimensions": dim,
                "model_id": BEDROCK_SETTINGS.embedding_model_id,
            }
        )
    except (BedrockConfigurationError, BedrockInferenceError) as exc:
        errors.append(f"Embedding probe failed: {exc}")
        steps.append({"name": "embedding_probe", "ok": False})
        return {"ok": False, "steps": steps, "errors": errors}

    # Step 3–4 — retriever construction + search
    try:
        t0 = time.perf_counter()
        retriever = KnowledgeRetriever(provider=provider)
        steps.append(
            {
                "name": "vector_store_load",
                "ok": True,
                "corpus": retriever.corpus,
            }
        )
    except RuntimeError as exc:
        errors.append(f"Vector store load failed: {exc}")
        steps.append({"name": "vector_store_load", "ok": False, "corpus": RAG_SETTINGS.corpus_name})
        return {"ok": False, "steps": steps, "errors": errors}

    try:
        t0 = time.perf_counter()
        hits = retriever.search(retrieval_probe_query, top_k=3)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if not hits:
            errors.append("Retrieval probe returned zero chunks.")
        steps.append(
            {
                "name": "retrieval_probe",
                "ok": bool(hits),
                "latency_ms": elapsed_ms,
                "chunks_returned": len(hits),
                "top_score": float(hits[0].score) if hits else None,
            }
        )
    except (BedrockInferenceError, RuntimeError) as exc:
        errors.append(f"Retrieval probe failed: {exc}")
        steps.append({"name": "retrieval_probe", "ok": False})
        return {"ok": False, "steps": steps, "errors": errors}

    ok = not errors and all(s.get("ok") for s in steps if "ok" in s)
    return {
        "ok": ok,
        "steps": steps,
        "errors": errors,
        "chat_model_id": BEDROCK_SETTINGS.chat_model_id,
        "embedding_model_id": BEDROCK_SETTINGS.embedding_model_id,
        "corpus_name": RAG_SETTINGS.corpus_name,
    }
