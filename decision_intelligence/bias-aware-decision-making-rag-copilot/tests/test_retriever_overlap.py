"""Unit tests for overlap-aware filtering helpers in ``rag.retriever``."""

from __future__ import annotations

import unittest

from rag.retriever import KnowledgeRetriever


def _bare_retriever() -> KnowledgeRetriever:
    """Construct without running ``__init__`` (no vector store required)."""
    return KnowledgeRetriever.__new__(KnowledgeRetriever)


class ChunkOverlapTests(unittest.TestCase):
    def test_same_source_adjacent_indices_overlap_within_radius(self) -> None:
        r = _bare_retriever()
        r.metadata = [
            {"source": "BookA", "chunk_index": 0},
            {"source": "BookA", "chunk_index": 1},
            {"source": "BookB", "chunk_index": 0},
        ]
        self.assertTrue(r._chunk_overlap(0, 1, radius=2))
        self.assertFalse(r._chunk_overlap(0, 2, radius=2))

    def test_different_source_no_overlap(self) -> None:
        r = _bare_retriever()
        r.metadata = [
            {"source": "A", "chunk_index": 0},
            {"source": "B", "chunk_index": 0},
        ]
        self.assertFalse(r._chunk_overlap(0, 1, radius=10))


class OverlapGreedyFilterTests(unittest.TestCase):
    def test_greedy_prefers_spread_then_backfills(self) -> None:
        r = _bare_retriever()
        r.metadata = [{"source": "S", "chunk_index": i} for i in range(20)]
        # Rows 0,1,2 are chunk_index 0,1,2 — with radius 1, keep 0 skip 1 keep 2? 2 overlaps 1? |2-1|=1 <=1 overlaps chunk 1 not in kept. kept=[0], try 1 overlaps 0, skip, try 2 overlaps 1? not in kept - chunk 2 vs kept 0: |2-0|=2 > radius 1 -> no overlap. keep 2. kept=[0,2]
        ordered = [0, 1, 2, 5]
        kept = r._filter_overlap_greedy(ordered, limit=3, radius=1)
        self.assertEqual(len(kept), 3)
        self.assertIn(0, kept)
        self.assertNotIn(1, kept)  # overlaps 0
        self.assertIn(2, kept)

    def test_backfill_when_pool_too_tight(self) -> None:
        r = _bare_retriever()
        r.metadata = [{"source": "S", "chunk_index": 0}] * 3  # malformed dup rows — still test backfill path
        ordered = [0, 1, 2]
        kept = r._filter_overlap_greedy(ordered, limit=3, radius=0)
        # radius 0: overlap only if same chunk_index AND same source — all chunk_index 0, rows overlap each other
        # First kept 0; 1 overlaps 0 (same ci); skip; 2 skip; backfill adds 1,2
        self.assertEqual(kept, [0, 1, 2])


class NeighborSnippetsTests(unittest.TestCase):
    def test_neighbors_same_source_only(self) -> None:
        r = _bare_retriever()
        r.metadata = [
            {"source": "B", "chunk_index": 0, "text": "zero"},
            {"source": "B", "chunk_index": 1, "text": "one text here"},
            {"source": "B", "chunk_index": 2, "text": "two"},
        ]
        r._faiss_idx_by_source_chunk = {("B", 0): 0, ("B", 1): 1, ("B", 2): 2}
        blocks = r._neighbor_snippets(1, neighbor_span=1, max_chars=50)
        self.assertEqual(len(blocks), 2)
        deltas = {int(b["chunk_index_delta"]) for b in blocks}
        self.assertEqual(deltas, {-1, 1})


if __name__ == "__main__":
    unittest.main()
