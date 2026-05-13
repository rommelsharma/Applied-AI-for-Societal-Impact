"""
Centralised application settings.

This is the only module that reads environment variables. Every other module
imports ``BEDROCK_SETTINGS`` and ``RAG_SETTINGS`` from here, so rotating a
region, swapping a model, or changing a retrieval default is a single-file
change. ``.env`` is loaded automatically via ``python-dotenv``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load ``.env`` early so any subsequent ``os.getenv`` call sees its values.
load_dotenv()


@dataclass(frozen=True)
class BedrockSettings:
    """Frozen Bedrock configuration consumed across the project.

    The defaults match ``.env.example`` and reflect the production intent
    (Claude Sonnet 4.5 for chat, Titan v2 at 1024 dimensions for embeddings,
    ``us-west-2`` region). Environment variables override defaults so the
    same code runs unchanged across local dev, CI, and deployed envs.
    """

    region: str = os.getenv("BEDROCK_REGION", "us-west-2")
    chat_model_id: str = os.getenv(
        "BEDROCK_MODEL_ID",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    embedding_model_id: str = os.getenv(
        "BEDROCK_EMBEDDING_MODEL_ID",
        "amazon.titan-embed-text-v2:0",
    )
    endpoint_url: str | None = os.getenv("BEDROCK_ENDPOINT_URL")
    embedding_dimensions: int = int(os.getenv("BEDROCK_EMBEDDING_DIMENSIONS", "1024"))

    @property
    def has_api_key(self) -> bool:
        """True when a Bedrock bearer token is configured.

        Used by ``BedrockProvider`` to fail fast at construction time instead
        of producing an opaque boto3 error deep inside a request path.
        """
        return bool(os.getenv("AWS_BEARER_TOKEN_BEDROCK"))


@dataclass(frozen=True)
class RAGSettings:
    """Retrieval and ranking configuration.

    Captures every knob that influences the retrieval path so that index
    versions (e.g. ``public`` vs ``private``) can be A/B tested under
    identical retrieval settings.
    """

    corpus_name: str = os.getenv("CORPUS_NAME", "public")
    chunk_profile: str = os.getenv("CHUNK_PROFILE", "auto")
    default_top_k: int = int(os.getenv("RAG_TOP_K", "8"))
    mmr_lambda: float = float(os.getenv("RAG_MMR_LAMBDA", "0.7"))
    reranker: str = os.getenv("RAG_RERANKER", "none")
    reranker_model_id: str = os.getenv(
        "RAG_RERANKER_MODEL_ID",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    reranker_candidates: int = int(os.getenv("RAG_RERANKER_CANDIDATES", "24"))
    enrichment_use_llm: bool = os.getenv("ENRICHMENT_USE_LLM", "false").lower() in {"1", "true", "yes"}
    # Index-time embedding: mean-pool vectors of sentence-centred windows (±N sentences)
    # for richer retrieval semantics. Query embeddings stay full user text (retriever).
    embed_sentence_windows: bool = os.getenv("RAG_EMBED_SENTENCE_WINDOWS", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    embed_sentence_radius: int = int(os.getenv("RAG_EMBED_SENTENCE_RADIUS", "3"))
    # 0 = embed every sentence-centred window (most Bedrock calls). Set e.g. 20 to subsample evenly.
    embed_max_windows_per_chunk: int = int(os.getenv("RAG_EMBEDDING_MAX_WINDOWS", "0"))


# Single shared instances imported by every module that needs them.
BEDROCK_SETTINGS = BedrockSettings()
RAG_SETTINGS = RAGSettings()
