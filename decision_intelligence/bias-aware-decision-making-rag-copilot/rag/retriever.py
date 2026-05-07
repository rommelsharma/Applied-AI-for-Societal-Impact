"""
Knowledge-base retrieval utilities backed by a FAISS vector index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from app.services.bedrock_provider import BedrockProvider
from shared_components.settings import BEDROCK_SETTINGS
from shared_components.utilities.path_utils import get_vector_store_dir


@dataclass
class RetrievalResult:
    id: str
    source: str
    author: str
    summary: str
    text: str
    concepts: list[str]
    decision_domains: list[str]
    score: float
    chunk_index: int


class KnowledgeRetriever:
    def __init__(self, provider: BedrockProvider | None = None):
        self.provider = provider or BedrockProvider()
        vector_store_dir = get_vector_store_dir()
        self.index = None
        self.embeddings = None

        index_file = vector_store_dir / "knowledge.index"
        embeddings_file = vector_store_dir / "embeddings.npy"

        try:
            import faiss
        except ImportError:
            faiss = None

        if faiss is not None and index_file.exists():
            self.index = faiss.read_index(str(index_file))
        elif embeddings_file.exists():
            self.embeddings = np.load(embeddings_file).astype(np.float32)
        else:
            raise RuntimeError(
                "No vector store artifacts found. Run data_pipeline/build_vector_index.py first."
            )

        with (vector_store_dir / "index_metadata.json").open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        preferred_concepts: list[str] | None = None,
        preferred_domains: list[str] | None = None,
    ) -> list[RetrievalResult]:
        query_vector = np.asarray(
            [self.provider.embed_text(query)],
            dtype=np.float32,
        )

        limit = max(top_k or BEDROCK_SETTINGS.default_top_k, 1)
        if self.index is not None:
            scores, indices = self.index.search(query_vector, limit * 3)
        else:
            similarity_scores = self.embeddings @ query_vector[0]
            candidate_indices = np.argsort(-similarity_scores)[: limit * 3]
            scores = np.asarray([similarity_scores[candidate_indices]], dtype=np.float32)
            indices = np.asarray([candidate_indices], dtype=np.int64)

        results: list[RetrievalResult] = []

        concept_filter = set(preferred_concepts or [])
        domain_filter = set(preferred_domains or [])

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            chunk = self.metadata[idx]

            if concept_filter and not concept_filter.intersection(chunk.get("concepts", [])):
                continue

            if domain_filter and not domain_filter.intersection(
                chunk.get("decision_domains", [])
            ):
                continue

            results.append(
                RetrievalResult(
                    id=chunk["id"],
                    source=chunk["source"],
                    author=chunk["author"],
                    summary=chunk["summary"],
                    text=chunk["text"],
                    concepts=chunk.get("concepts", []),
                    decision_domains=chunk.get("decision_domains", []),
                    score=float(score),
                    chunk_index=chunk.get("chunk_index", -1),
                )
            )

            if len(results) >= limit:
                break

        if results:
            return results

        # Fall back to unfiltered semantic retrieval if filters were too restrictive.
        if self.index is not None:
            fallback_scores, fallback_indices = self.index.search(query_vector, limit)
        else:
            similarity_scores = self.embeddings @ query_vector[0]
            candidate_indices = np.argsort(-similarity_scores)[:limit]
            fallback_scores = np.asarray([similarity_scores[candidate_indices]], dtype=np.float32)
            fallback_indices = np.asarray([candidate_indices], dtype=np.int64)
        fallback_results: list[RetrievalResult] = []

        for score, idx in zip(fallback_scores[0], fallback_indices[0]):
            if idx < 0:
                continue

            chunk = self.metadata[idx]
            fallback_results.append(
                RetrievalResult(
                    id=chunk["id"],
                    source=chunk["source"],
                    author=chunk["author"],
                    summary=chunk["summary"],
                    text=chunk["text"],
                    concepts=chunk.get("concepts", []),
                    decision_domains=chunk.get("decision_domains", []),
                    score=float(score),
                    chunk_index=chunk.get("chunk_index", -1),
                )
            )

        return fallback_results
