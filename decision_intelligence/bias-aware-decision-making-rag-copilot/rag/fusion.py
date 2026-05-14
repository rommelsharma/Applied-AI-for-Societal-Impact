"""Reciprocal Rank Fusion (RRF) for hybrid retrieval lists."""

from __future__ import annotations


def reciprocal_rank_fusion(rank_lists: list[list[int]], *, k: int = 60) -> tuple[list[int], dict[int, float]]:
    """Merge multiple ranked id lists into one ordering by RRF score.

    Returns ``(ordered_indices, rrf_score_by_index)``.
    """
    scores: dict[int, float] = {}
    for ranked in rank_lists:
        if not ranked:
            continue
        for rank, idx in enumerate(ranked):
            if idx < 0:
                continue
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    ordered = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    return ordered, scores
