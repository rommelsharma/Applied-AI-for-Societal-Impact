"""Lightweight query intent classification (fail-open to ``recall``)."""

from __future__ import annotations

import re


def classify_query_intent(query: str) -> dict:
    """Return ``intent`` in ``recall|comparison|synthesis|mitigation`` and ``mmr_lambda`` hint.

    Rule-based default avoids extra Bedrock calls. When ``RAG_QUERY_CLASSIFICATION=haiku``,
    callers may extend this module later with a Haiku call.
    """
    q = (query or "").lower()
    if re.search(r"\b(compare|versus|vs\.?|difference between|contrasted with)\b", q):
        return {"intent": "comparison", "mmr_lambda": 0.65}
    if re.search(r"\b(how to|mitigat|reduce bias|debias|prevent|avoid|fix|remedy)\b", q):
        return {"intent": "mitigation", "mmr_lambda": 0.75}
    if re.search(r"\b(synthes|overall|big picture|integrate|across books|together)\b", q):
        return {"intent": "synthesis", "mmr_lambda": 0.6}
    return {"intent": "recall", "mmr_lambda": 0.7}
