"""
Application settings and environment-driven configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class BedrockSettings:
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
    default_top_k: int = int(os.getenv("RAG_TOP_K", "6"))

    @property
    def has_api_key(self) -> bool:
        return bool(os.getenv("AWS_BEARER_TOKEN_BEDROCK"))


BEDROCK_SETTINGS = BedrockSettings()
