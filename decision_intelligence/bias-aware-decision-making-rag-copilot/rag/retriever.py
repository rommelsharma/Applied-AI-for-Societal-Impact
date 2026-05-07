"""
Knowledge-base retrieval layer.

Loads the FAISS index (or a numpy fallback) plus the aligned chunk metadata
produced by ``data_pipeline/build_vector_index.py``, and exposes a single
``search`` entry point that the bias detector calls at runtime.

Design choices that matter:
    * FAISS ``IndexFlatIP`` with normalised Titan v2 embeddings is
      mathematically equivalent to cosine similarity, with zero approximation
      error at the project's current corpus size.
    * Concept and decision-domain filters are applied AFTER an initial broad
      semantic search; if filters strip everything the retriever falls back
      to unfiltered semantic results so a query never returns empty.
    * MMR (Maximal Marginal Relevance) re-ordering diversifies the top-k so
      a single book / chapter cannot monopolise the result set.
    * An optional reranker stage (Claude Haiku as relevance judge) sits in
      front of the final cut. Off by default; toggled via env.

The retriever is corpus-aware: pass ``corpus="private"`` at construction to
target an alternative index (used by ``compare_versions.py`` and any future
public-vs-private demo orchestration).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from app.services.bedrock_provider import BedrockProvider
from shared_components.settings import BEDROCK_SETTINGS, RAG_SETTINGS
from shared_components.utilities.path_utils import get_corpus_name, get_vector_store_dir


@dataclass
class RetrievalResult:
    """Single retrieved chunk together with full source attribution and similarity score."""

    id: str
    source: str
    author: str
    summary: str
    text: str
    concepts: list[str]
    decision_domains: list[str]
    score: float
    chunk_index: int
    chapter_title: str = ""
    passage_type: str = "unknown"
    decision_phase: str = "unknown"


class KnowledgeRetriever:
    """FAISS-backed retriever with MMR diversification and optional reranking.

    Construction loads the index and metadata once. The hot path
    (``search``) only embeds the query and runs a similarity lookup, so the
    same instance can be reused across many scenarios cheaply.
    """

    def __init__(self, provider: BedrockProvider | None = None, corpus: str | None = None):
        # Allow the caller to inject a provider so a single Bedrock client can
        # be shared across embedding and chat calls in a single request.
        self.provider = provider or BedrockProvider()
        self.corpus = get_corpus_name(corpus)
        vector_store_dir = get_vector_store_dir(self.corpus)
        self.index = None
        self.embeddings: np.ndarray | None = None

        index_file = vector_store_dir / "knowledge.index"
        embeddings_file = vector_store_dir / "embeddings.npy"
        metadata_file = vector_store_dir / "index_metadata.json"

        # FAISS is the preferred backend; numpy is a graceful fallback so the
        # system still works on environments where the wheel is unavailable.
        try:
            import faiss
        except ImportError:
            faiss = None

        # ALWAYS load the embeddings matrix when present, even when FAISS is
        # available - MMR needs the raw vectors to compute pairwise similarity
        # between candidates.
        if embeddings_file.exists():
            self.embeddings = np.load(embeddings_file).astype(np.float32)

        if faiss is not None and index_file.exists():
            self.index = faiss.read_index(str(index_file))
        elif self.embeddings is None:
            raise RuntimeError(
                f"No vector store artefacts found for corpus '{self.corpus}'. "
                f"Run `python data_pipeline/build_vector_index.py --corpus {self.corpus}` first."
            )

        # ``index_metadata.json`` is the chunk payload, in the SAME row order
        # as the FAISS index / numpy matrix - this alignment is what lets us
        # look up source metadata by integer index.
        with metadata_file.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query and shape it for FAISS / numpy similarity ops."""
        return np.asarray([self.provider.embed_text(query)], dtype=np.float32)

    def _semantic_candidates(self, query_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Pull ``k`` raw candidates by inner-product similarity.

        Uses FAISS when available; falls back to a dot-product on the raw
        embeddings matrix otherwise. Always returns ``(scores, indices)``
        with shape ``(1, k)`` so downstream code is shape-uniform.
        """
        if self.index is not None:
            scores, indices = self.index.search(query_vector, k)
            return scores, indices
        assert self.embeddings is not None
        similarity_scores = self.embeddings @ query_vector[0]
        candidate_indices = np.argsort(-similarity_scores)[:k]
        return (
            np.asarray([similarity_scores[candidate_indices]], dtype=np.float32),
            np.asarray([candidate_indices], dtype=np.int64),
        )

    def _passes_filters(
        self,
        chunk: dict,
        concept_filter: set[str],
        domain_filter: set[str],
    ) -> bool:
        """True iff ``chunk`` satisfies the optional concept/domain filters."""
        if concept_filter and not concept_filter.intersection(chunk.get("concepts", [])):
            return False
        if domain_filter and not domain_filter.intersection(chunk.get("decision_domains", [])):
            return False
        return True

    def _build_result(self, chunk: dict, score: float) -> RetrievalResult:
        """Wrap a metadata row plus its similarity into a ``RetrievalResult``."""
        return RetrievalResult(
            id=chunk["id"],
            source=chunk["source"],
            author=chunk["author"],
            summary=chunk["summary"],
            text=chunk["text"],
            concepts=chunk.get("concepts", []),
            decision_domains=chunk.get("decision_domains", []),
            score=float(score),
            chunk_index=chunk.get("chunk_index", -1),
            chapter_title=chunk.get("chapter_title", ""),
            passage_type=chunk.get("passage_type", "unknown"),
            decision_phase=chunk.get("decision_phase", "unknown"),
        )

    def _apply_mmr(
        self,
        candidate_indices: Iterable[int],
        candidate_scores: dict[int, float],
        query_vector: np.ndarray,
        k: int,
        lambda_value: float,
    ) -> list[int]:
        """Diversify candidates with Maximal Marginal Relevance.

        MMR picks one candidate at a time, balancing relevance to the query
        against dissimilarity from already-picked candidates::

            score(c) = λ * sim(c, query) - (1 - λ) * max_{p∈picked} sim(c, p)

        ``λ → 1`` recovers pure relevance; ``λ → 0`` is pure diversity. The
        default of 0.7 keeps relevance dominant while still discouraging the
        retriever from returning four near-identical chunks from the same
        chapter.

        Falls back to relevance-only ordering when the embeddings matrix is
        not loaded (rare; only when neither FAISS nor numpy embeddings exist).
        """
        candidates = [idx for idx in candidate_indices if idx >= 0]
        if not candidates:
            return []
        if self.embeddings is None:
            return sorted(candidates, key=lambda i: candidate_scores.get(i, 0.0), reverse=True)[:k]

        query_unit = query_vector[0]
        picked: list[int] = []
        remaining = list(candidates)

        while remaining and len(picked) < k:
            best_idx = None
            best_score = -float("inf")
            for idx in remaining:
                relevance = candidate_scores.get(idx, float(self.embeddings[idx] @ query_unit))
                if picked:
                    diversity = max(float(self.embeddings[idx] @ self.embeddings[p]) for p in picked)
                else:
                    diversity = 0.0
                score = lambda_value * relevance - (1.0 - lambda_value) * diversity
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            picked.append(best_idx)
            remaining.remove(best_idx)

        return picked

    def _rerank_with_llm(
        self,
        query: str,
        candidates: list[tuple[float, int]],
        top_k: int,
    ) -> list[tuple[float, int]]:
        """Optional LLM-as-judge reranker over the FAISS top candidates.

        Disabled by default. When ``RAG_RERANKER == 'claude_haiku'`` the
        candidates are presented to Claude Haiku which returns a relevance
        score between 0 and 1 per candidate; results are then re-sorted by
        the LLM scores. Falls back to the original ordering if the LLM
        response is malformed or Bedrock raises - retrieval must never break
        because reranking went sideways.
        """
        if RAG_SETTINGS.reranker != "claude_haiku":
            return candidates[:top_k]

        try:
            blocks = []
            for rank, (_, idx) in enumerate(candidates, start=1):
                chunk = self.metadata[idx]
                blocks.append(
                    f"[{rank}] source={chunk.get('source', '')} concepts={chunk.get('concepts', [])}\n"
                    f"summary={chunk.get('summary', '')}"
                )

            user_prompt = (
                f"Query:\n{query}\n\n"
                f"Candidates:\n{chr(10).join(blocks)}\n\n"
                "For each candidate, return a JSON array of objects: "
                '[{"rank": 1, "score": 0.0-1.0}, ...] '
                "where score is the candidate's relevance to the query. JSON only."
            )

            from app.services.bedrock_provider import BedrockProvider

            provider = BedrockProvider()
            previous_chat_model = BEDROCK_SETTINGS.chat_model_id
            # Use the configured reranker model id without disturbing global settings.
            provider.client.meta.config  # access guarantees client materialised
            result = provider.client.converse(
                modelId=RAG_SETTINGS.reranker_model_id,
                system=[{"text": "You are a strict relevance judge. Return JSON only."}],
                messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                inferenceConfig={"temperature": 0.0, "maxTokens": 600},
            )
            text = "\n".join(
                block.get("text", "") for block in result["output"]["message"]["content"] if "text" in block
            )
            parsed = json.loads(text[text.find("[") : text.rfind("]") + 1])
            score_by_rank = {int(item["rank"]): float(item["score"]) for item in parsed}

            rescored: list[tuple[float, int]] = []
            for rank, (_, idx) in enumerate(candidates, start=1):
                rescored.append((score_by_rank.get(rank, 0.0), idx))
            rescored.sort(key=lambda pair: pair[0], reverse=True)
            return rescored[:top_k]
        except Exception:  # pragma: no cover - reranker must never break retrieval
            return candidates[:top_k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        preferred_concepts: list[str] | None = None,
        preferred_domains: list[str] | None = None,
        mmr_lambda: float | None = None,
        reranker_candidates: int | None = None,
    ) -> list[RetrievalResult]:
        """Return the top-k chunks most relevant to ``query``.

        Pipeline:
            1. Embed the query via Bedrock Titan v2.
            2. Pull a wide candidate set (``top_k * 3`` or
               ``RAG_RERANKER_CANDIDATES`` when reranking is enabled).
            3. Apply concept and decision-domain filters.
            4. Rerank with the configured reranker (no-op by default).
            5. Diversify with MMR using ``mmr_lambda``.
            6. Fall back to unfiltered semantic results if filtering left
               nothing - the caller must always get usable context.
        """
        limit = max(top_k or RAG_SETTINGS.default_top_k, 1)
        candidate_pool_size = max(
            limit * 3,
            reranker_candidates or (RAG_SETTINGS.reranker_candidates if RAG_SETTINGS.reranker != "none" else 0),
        )

        query_vector = self._embed_query(query)
        scores, indices = self._semantic_candidates(query_vector, candidate_pool_size)

        concept_filter = set(preferred_concepts or [])
        domain_filter = set(preferred_domains or [])
        candidate_scores: dict[int, float] = {}
        filtered_pairs: list[tuple[float, int]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.metadata[idx]
            candidate_scores[int(idx)] = float(score)
            if not self._passes_filters(chunk, concept_filter, domain_filter):
                continue
            filtered_pairs.append((float(score), int(idx)))

        # Filter fallback: never return empty.
        if not filtered_pairs:
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                filtered_pairs.append((float(score), int(idx)))

        # Optional reranker stage. No-op when reranker == "none".
        ranked = self._rerank_with_llm(
            query=query,
            candidates=filtered_pairs,
            top_k=max(limit, candidate_pool_size // 2),
        )

        # Diversify the final cut with MMR so several books/chapters can
        # appear in the result set instead of one source dominating.
        ordered_indices = self._apply_mmr(
            candidate_indices=[idx for _, idx in ranked],
            candidate_scores={idx: score for score, idx in ranked},
            query_vector=query_vector,
            k=limit,
            lambda_value=mmr_lambda if mmr_lambda is not None else RAG_SETTINGS.mmr_lambda,
        )

        results: list[RetrievalResult] = []
        for idx in ordered_indices:
            chunk = self.metadata[idx]
            score = candidate_scores.get(idx, 0.0)
            results.append(self._build_result(chunk, score))

        return results
