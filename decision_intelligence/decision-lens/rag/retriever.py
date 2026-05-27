"""
Knowledge-base retrieval layer.

Hybrid **dense + BM25 + RRF** when ``bm25_index.pkl`` is present and
``RAG_BM25_ENABLED`` is true; otherwise dense-only (legacy). Index artefacts
live under ``processed/index`` when built with the v4 pipeline, with fallback
to ``vector_store/``.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from app.services.bedrock_provider import BedrockProvider
from rag.fusion import reciprocal_rank_fusion
from shared_components.settings import BEDROCK_SETTINGS, RAG_SETTINGS
from shared_components.utilities.path_utils import (
    get_corpus_name,
    get_vector_store_dir,
    resolve_vector_index_dir,
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


@dataclass
class RetrievalResult:
    """Single retrieved chunk with attribution, hybrid scores, and ontology fields."""

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
    neighbor_blocks: list[dict[str, str]] = field(default_factory=list)
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    retrieval_method: str = "dense"
    biases: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)


class KnowledgeRetriever:
    """FAISS + optional BM25/RRF, MMR, optional reranker, overlap filter."""

    def __init__(self, provider: BedrockProvider | None = None, corpus: str | None = None):
        self.provider = provider or BedrockProvider()
        self.corpus = get_corpus_name(corpus)
        self._index_dir = resolve_vector_index_dir(self.corpus)
        self.index = None
        self.embeddings: np.ndarray | None = None
        self._bm25 = None

        index_file = self._index_dir / "knowledge.index"
        embeddings_file = self._index_dir / "embeddings.npy"
        metadata_file = self._index_dir / "index_metadata.json"

        try:
            import faiss
        except ImportError:
            faiss = None

        if embeddings_file.exists():
            self.embeddings = np.load(embeddings_file).astype(np.float32)

        if faiss is not None and index_file.exists():
            self.index = faiss.read_index(str(index_file))
        elif self.embeddings is None:
            raise RuntimeError(
                f"No vector store artefacts found for corpus '{self.corpus}'. "
                f"Run `python data_pipeline/build_vector_index.py --corpus {self.corpus}` first."
            )

        with metadata_file.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

        self._faiss_idx_by_source_chunk: dict[tuple[str, int], int] = {}
        for row_i, chunk in enumerate(self.metadata):
            src = chunk.get("source") or ""
            ci = chunk.get("chunk_index")
            if isinstance(ci, int) and ci >= 0 and src:
                self._faiss_idx_by_source_chunk[(src, ci)] = row_i

        self._load_bm25()

    def _bm25_candidates_path(self) -> tuple:
        primary = self._index_dir / "bm25_index.pkl"
        if primary.is_file():
            return primary, "processed_index"
        legacy = get_vector_store_dir(self.corpus) / "bm25_index.pkl"
        if legacy.is_file():
            return legacy, "legacy_vector_store"
        return primary, "none"

    def _load_bm25(self) -> None:
        if not RAG_SETTINGS.bm25_enabled:
            return
        path, _origin = self._bm25_candidates_path()
        if not path.is_file():
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        docs = payload.get("tokenized_docs")
        if isinstance(docs, list) and docs:
            self._bm25 = BM25Okapi(docs)

    def _bm25_ranked_indices(self, query: str, top_n: int) -> list[int]:
        if self._bm25 is None or top_n <= 0:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = np.argsort(-scores)[:top_n]
        return [int(i) for i in order]

    def _chunk_overlap(self, row_a: int, row_b: int, radius: int) -> bool:
        if row_a < 0 or row_b < 0 or row_a == row_b:
            return row_a == row_b
        ca = self.metadata[row_a]
        cb = self.metadata[row_b]
        if (ca.get("source") or "") != (cb.get("source") or ""):
            return False
        ia = ca.get("chunk_index")
        ib = cb.get("chunk_index")
        if not isinstance(ia, int) or not isinstance(ib, int) or ia < 0 or ib < 0:
            return False
        return abs(ia - ib) <= radius

    def _filter_overlap_greedy(self, ordered: list[int], limit: int, radius: int) -> list[int]:
        kept: list[int] = []
        for idx in ordered:
            if len(kept) >= limit:
                break
            if any(self._chunk_overlap(idx, k, radius) for k in kept):
                continue
            kept.append(idx)
        if len(kept) < limit:
            for idx in ordered:
                if len(kept) >= limit:
                    break
                if idx in kept:
                    continue
                kept.append(idx)
        return kept[:limit]

    def _neighbor_snippets(self, primary_row: int, neighbor_span: int, max_chars: int) -> list[dict[str, str]]:
        if neighbor_span <= 0 or max_chars <= 0:
            return []
        chunk = self.metadata[primary_row]
        src = chunk.get("source") or ""
        ci = chunk.get("chunk_index")
        if not isinstance(ci, int) or ci < 0 or not src:
            return []
        blocks: list[dict[str, str]] = []
        for delta in range(-neighbor_span, neighbor_span + 1):
            if delta == 0:
                continue
            nci = ci + delta
            row_j = self._faiss_idx_by_source_chunk.get((src, nci))
            if row_j is None:
                continue
            meta = self.metadata[row_j]
            text = (meta.get("text") or "").strip()
            excerpt = text[:max_chars] if text else ""
            if not excerpt:
                continue
            blocks.append(
                {
                    "chunk_id": str(meta.get("id", "")),
                    "chunk_index_delta": str(delta),
                    "excerpt": excerpt,
                }
            )
        blocks.sort(key=lambda b: int(b.get("chunk_index_delta", "0")))
        return blocks

    def _embed_query(self, query: str) -> np.ndarray:
        return np.asarray([self.provider.embed_text(query)], dtype=np.float32)

    def _semantic_candidates(self, query_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
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

    def _symbolic_concept_bag(self, chunk: dict) -> set[str]:
        bag: set[str] = set()
        for key in (
            "concepts",
            "biases",
            "failure_modes",
            "interventions",
            "related_concepts",
            "mitigated_by_concepts",
            "cognitive_mechanisms",
        ):
            for x in chunk.get(key) or []:
                if isinstance(x, str):
                    bag.add(x)
        return bag

    def _passes_filters(
        self,
        chunk: dict,
        concept_filter: set[str],
        domain_filter: set[str],
    ) -> bool:
        if concept_filter and not self._symbolic_concept_bag(chunk).intersection(concept_filter):
            return False
        if domain_filter and not domain_filter.intersection(chunk.get("decision_domains", [])):
            return False
        return True

    def _build_result(
        self,
        chunk: dict,
        score: float,
        *,
        neighbor_blocks: list[dict[str, str]] | None = None,
        bm25_score: float = 0.0,
        rrf_score: float = 0.0,
        retrieval_method: str = "dense",
    ) -> RetrievalResult:
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
            neighbor_blocks=list(neighbor_blocks or []),
            bm25_score=float(bm25_score),
            rrf_score=float(rrf_score),
            retrieval_method=retrieval_method,
            biases=list(chunk.get("biases") or []),
            failure_modes=list(chunk.get("failure_modes") or []),
            interventions=list(chunk.get("interventions") or []),
        )

    def _apply_mmr(
        self,
        candidate_indices: Iterable[int],
        candidate_scores: dict[int, float],
        query_vector: np.ndarray,
        k: int,
        lambda_value: float,
    ) -> list[int]:
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

            provider = BedrockProvider()
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
        except Exception:  # pragma: no cover
            return candidates[:top_k]

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
        limit = max(top_k or RAG_SETTINGS.default_top_k, 1)
        overlap_mult = max(1, int(RAG_SETTINGS.overlap_candidate_pool_multiplier)) if RAG_SETTINGS.overlap_filter else 1
        base_pool = max(
            limit * 3,
            reranker_candidates or (RAG_SETTINGS.reranker_candidates if RAG_SETTINGS.reranker != "none" else 0),
        )
        candidate_pool_size = max(base_pool * overlap_mult, limit * 3)

        query_vector = self._embed_query(query)
        scores, indices = self._semantic_candidates(query_vector, candidate_pool_size)

        dense_ranked: list[int] = []
        dense_scores: dict[int, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            ii = int(idx)
            dense_ranked.append(ii)
            dense_scores[ii] = float(score)

        bm25_ranked = self._bm25_ranked_indices(query, RAG_SETTINGS.bm25_top_k) if self._bm25 else []

        if bm25_ranked and RAG_SETTINGS.bm25_enabled:
            fused_order, rrf_scores = reciprocal_rank_fusion(
                [dense_ranked, bm25_ranked],
                k=RAG_SETTINGS.rrf_k,
            )
            retrieval_mode = "hybrid"
        else:
            fused_order = list(dense_ranked)
            rrf_scores = {i: dense_scores.get(i, 0.0) for i in fused_order}
            retrieval_mode = "dense"

        concept_filter = set(preferred_concepts or [])
        domain_filter = set(preferred_domains or [])

        candidate_scores: dict[int, float] = {}
        for idx in fused_order:
            qv = query_vector[0]
            dense_part = dense_scores.get(idx)
            if dense_part is not None:
                rel = dense_part
            elif self.embeddings is not None:
                rel = float(self.embeddings[idx] @ qv)
            else:
                rel = 0.0
            rrf_part = rrf_scores.get(idx, 0.0)
            candidate_scores[idx] = max(rel, 0.15 * rrf_part)

        filtered_pairs: list[tuple[float, int]] = []
        for idx in fused_order:
            if idx < 0 or idx >= len(self.metadata):
                continue
            chunk = self.metadata[idx]
            if not self._passes_filters(chunk, concept_filter, domain_filter):
                continue
            filtered_pairs.append((candidate_scores[idx], idx))

        if len(filtered_pairs) < RAG_SETTINGS.ontology_filter_min_results and concept_filter:
            filtered_pairs = []
            for idx in fused_order:
                if idx < 0 or idx >= len(self.metadata):
                    continue
                chunk = self.metadata[idx]
                if not self._passes_filters(chunk, set(), domain_filter):
                    continue
                filtered_pairs.append((candidate_scores[idx], idx))

        if not filtered_pairs:
            for idx in fused_order:
                if idx < 0 or idx >= len(self.metadata):
                    continue
                filtered_pairs.append((candidate_scores.get(idx, 0.0), idx))

        ranked = self._rerank_with_llm(
            query=query,
            candidates=filtered_pairs,
            top_k=max(limit, candidate_pool_size // 2),
        )

        ranked_indices = [idx for _, idx in ranked]
        mmr_k = limit
        if RAG_SETTINGS.overlap_filter:
            mmr_k = min(
                len(ranked_indices),
                max(limit, limit * max(2, int(RAG_SETTINGS.overlap_mmr_pool_multiplier))),
            )

        lam = mmr_lambda if mmr_lambda is not None else RAG_SETTINGS.mmr_lambda
        ordered_indices = self._apply_mmr(
            candidate_indices=ranked_indices,
            candidate_scores={idx: score for score, idx in ranked},
            query_vector=query_vector,
            k=mmr_k,
            lambda_value=lam,
        )

        if RAG_SETTINGS.overlap_filter:
            ordered_indices = self._filter_overlap_greedy(
                ordered_indices,
                limit,
                max(0, int(RAG_SETTINGS.overlap_chunk_radius)),
            )
        else:
            ordered_indices = ordered_indices[:limit]

        expand_n = max(0, int(RAG_SETTINGS.context_expand_neighbors))
        max_side = max(0, int(RAG_SETTINGS.expand_max_chars_per_side))

        bm25_raw: dict[int, float] = {}
        if self._bm25:
            toks = _tokenize(query)
            if toks:
                arr = self._bm25.get_scores(toks)
                for i, v in enumerate(arr):
                    bm25_raw[i] = float(v)

        results: list[RetrievalResult] = []
        for idx in ordered_indices:
            chunk = self.metadata[idx]
            score = candidate_scores.get(idx, 0.0)
            neighbors = (
                self._neighbor_snippets(idx, expand_n, max_side)
                if expand_n and max_side
                else []
            )
            b25 = bm25_raw.get(idx, 0.0)
            rr = rrf_scores.get(idx, 0.0)
            method = retrieval_mode if b25 or retrieval_mode == "hybrid" else "dense"
            results.append(
                self._build_result(
                    chunk,
                    score,
                    neighbor_blocks=neighbors,
                    bm25_score=b25,
                    rrf_score=rr,
                    retrieval_method=method,
                )
            )

        return results
