"""
Amazon Bedrock provider layer.

Encapsulates every Bedrock interaction the project needs:
    - chat completion via the Converse API (Claude Sonnet 4.5)
    - text embeddings via Amazon Titan Text Embeddings v2

All other modules talk to Bedrock through this class. Centralising access here
keeps the boto3 surface, error handling, and configuration in one place, and
makes swapping models or providers a single-file change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from shared_components.settings import BEDROCK_SETTINGS


class BedrockConfigurationError(RuntimeError):
    """Raised when required Bedrock configuration (e.g. credentials) is missing."""


class BedrockInferenceError(RuntimeError):
    """Raised when Bedrock returns an unrecoverable inference-time error."""


@dataclass
class LLMResult:
    """Container for a single chat completion response.

    Attributes:
        text: The concatenated text content extracted from the model's reply.
        raw_response: The full Bedrock response payload (kept for debugging/observability).
        usage: The token-usage dict returned by Bedrock, when available.
    """

    text: str
    raw_response: dict
    usage: dict


def _extract_text_from_content_blocks(content_blocks: list[dict]) -> str:
    """Flatten the Converse API ``content`` blocks into a single text string.

    Bedrock returns assistant content as a list of typed blocks (text, tool use, etc.).
    We currently only consume text blocks; this helper pulls them out and joins
    them so the caller can treat the response as plain text.
    """

    parts: list[str] = []

    for block in content_blocks:
        if "text" in block:
            parts.append(block["text"])

    return "\n".join(parts).strip()


class BedrockProvider:
    """Thin wrapper around the ``bedrock-runtime`` boto3 client.

    Exposes two operations the system actually needs:
        * :meth:`converse` - chat completion against the configured chat model.
        * :meth:`embed_text` - dense embedding for a single text input.

    Construction validates that credentials are configured so failures surface
    immediately at startup rather than deep inside a request path.
    """

    def __init__(self):
        # Fail fast if the bearer token / AWS credentials are not configured;
        # this avoids opaque boto3 errors much later in the request path.
        if not BEDROCK_SETTINGS.has_api_key:
            raise BedrockConfigurationError(
                "AWS_BEARER_TOKEN_BEDROCK is not set. "
                "Add it to your environment or .env before invoking Bedrock."
            )

        client_kwargs = {
            "service_name": "bedrock-runtime",
            "region_name": BEDROCK_SETTINGS.region,
        }

        # Allow an explicit endpoint override (useful for VPC endpoints / local proxies).
        if BEDROCK_SETTINGS.endpoint_url:
            client_kwargs["endpoint_url"] = BEDROCK_SETTINGS.endpoint_url

        self.client = boto3.client(**client_kwargs)

        # Auditable record: each materialised boto3 client is one logical Bedrock connection.
        try:
            from evaluation.connectivity import append_connectivity_log_line

            append_connectivity_log_line(
                True,
                f"bedrock_runtime_client_initialized region={BEDROCK_SETTINGS.region}",
            )
        except Exception:
            # Never break construction if logging fails (e.g. read-only cwd).
            pass

    def converse(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1800,
    ) -> LLMResult:
        """Run a single-turn chat completion against the configured chat model.

        Args:
            system_prompt: The system message that locks behaviour and output schema.
            user_prompt: The user-turn text containing the scenario and instructions.
            temperature: Sampling temperature; default ``0.1`` for near-deterministic output
                so evaluation runs are reproducible.
            max_tokens: Hard ceiling on response length; sized to fit the strict JSON schema
                used by the bias detector without truncation.

        Returns:
            ``LLMResult`` with parsed text, raw response payload, and usage metadata.

        Raises:
            BedrockInferenceError: Any boto3 / Bedrock failure is wrapped to give callers
                a single exception type to handle.
        """
        try:
            response = self.client.converse(
                modelId=BEDROCK_SETTINGS.chat_model_id,
                system=[{"text": system_prompt}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": user_prompt}],
                    }
                ],
                inferenceConfig={
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                },
            )
        except (ClientError, BotoCoreError) as exc:
            raise BedrockInferenceError(f"Bedrock Converse request failed: {exc}") from exc

        content = response["output"]["message"]["content"]
        text = _extract_text_from_content_blocks(content)

        return LLMResult(
            text=text,
            raw_response=response,
            usage=response.get("usage", {}),
        )

    def embed_text(self, text: str) -> list[float]:
        """Compute a dense embedding for a single piece of text.

        Uses the Titan Text Embeddings v2 model with ``normalize=true`` so that
        the resulting unit vectors can be compared with inner-product similarity
        (mathematically equivalent to cosine similarity), which is what the
        FAISS ``IndexFlatIP`` index expects.

        Args:
            text: The text to embed. Either a knowledge-base chunk (at indexing time)
                or a user query (at retrieval time).

        Returns:
            A list of floats of length ``BEDROCK_SETTINGS.embedding_dimensions``.

        Raises:
            BedrockInferenceError: Any boto3 / Bedrock failure is wrapped uniformly.
        """
        payload = json.dumps(
            {
                "inputText": text,
                "dimensions": BEDROCK_SETTINGS.embedding_dimensions,
                "normalize": True,
            }
        )

        try:
            response = self.client.invoke_model(
                modelId=BEDROCK_SETTINGS.embedding_model_id,
                body=payload,
                accept="application/json",
                contentType="application/json",
            )
        except (ClientError, BotoCoreError) as exc:
            raise BedrockInferenceError(f"Bedrock embedding request failed: {exc}") from exc

        response_body = json.loads(response["body"].read())
        return response_body["embedding"]
