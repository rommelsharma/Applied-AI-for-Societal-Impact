"""Unit tests for RRF fusion."""

from __future__ import annotations

import unittest

from rag.fusion import reciprocal_rank_fusion


class RRFusionTests(unittest.TestCase):
    def test_fusion_orders_union(self) -> None:
        ordered, scores = reciprocal_rank_fusion([[0, 1, 2], [2, 3, 0]], k=60)
        self.assertIn(0, ordered[:2])
        self.assertGreater(scores[2], 0.0)

    def test_single_list(self) -> None:
        ordered, scores = reciprocal_rank_fusion([[5, 4]], k=10)
        self.assertEqual(ordered, [5, 4])


if __name__ == "__main__":
    unittest.main()
