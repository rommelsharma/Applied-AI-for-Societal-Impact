"""
Bedrock-backed provider utilities for chat completion and embeddings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from shared_components.settings import BEDROCK_SETTINGS


class BedrockConfigurationError(RuntimeError):
    """Raised when required Bedrock configuration is missing."""


class BedrockInferenceError(RuntimeError):
    """Raised when Bedrock returns an unrecoverable inference error."""


@dataclass
class LLMResult:
    text: str
    raw_response: dict
    usage: dict


def _extract_text_from_content_blocks(content_blocks: list[dict]) -> str:
    parts: list[str] = []

    for block in content_blocks:
        if "text" in block:
            parts.append(block["text"])

    return "\n".join(parts).strip()


class BedrockProvider:
    def __init__(self):
        if not BEDROCK_SETTINGS.has_api_key:
            raise BedrockConfigurationError(
                "AWS_BEARER_TOKEN_BEDROCK is not set. "
                "Add it to your environment or .env before invoking Bedrock."
            )

        client_kwargs = {
            "service_name": "bedrock-runtime",
            "region_name": BEDROCK_SETTINGS.region,
        }

        if BEDROCK_SETTINGS.endpoint_url:
            client_kwargs["endpoint_url"] = BEDROCK_SETTINGS.endpoint_url

        self.client = boto3.client(**client_kwargs)

    def converse(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1800,
    ) -> LLMResult:
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
