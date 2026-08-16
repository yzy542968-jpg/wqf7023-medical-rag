from __future__ import annotations

import unittest

from medical_rag.retrieval.adaptive_retrieval import score_margin, select_adaptive_top1


class AdaptiveRetrievalTests(unittest.TestCase):
    def test_score_margin(self) -> None:
        self.assertAlmostEqual(score_margin([0.9, 0.7, 0.2]), 0.2)

    def test_agreement_keeps_shared_top1(self) -> None:
        result = select_adaptive_top1(
            base_case_ids=["a", "b"],
            base_scores=[0.9, 0.8],
            reranked_case_ids=["a", "b"],
            reranker_scores=[3.0, 1.0],
            reranker_margin_threshold=0.5,
            base_margin_threshold=0.1,
        )
        self.assertEqual(result.selected_case_id, "a")
        self.assertEqual(result.source, "agreement")

    def test_confident_reranker_overrides_uncertain_base(self) -> None:
        result = select_adaptive_top1(
            base_case_ids=["a", "b"],
            base_scores=[0.90, 0.88],
            reranked_case_ids=["b", "a"],
            reranker_scores=[4.0, 1.0],
            reranker_margin_threshold=1.0,
            base_margin_threshold=0.05,
        )
        self.assertEqual(result.selected_case_id, "b")
        self.assertEqual(result.source, "reranker")

    def test_low_confidence_can_abstain(self) -> None:
        result = select_adaptive_top1(
            base_case_ids=["a", "b"],
            base_scores=[0.80, 0.79],
            reranked_case_ids=["a", "b"],
            reranker_scores=[1.0, 0.99],
            reranker_margin_threshold=1.0,
            base_margin_threshold=0.05,
            minimum_base_score=0.9,
            minimum_selected_margin=0.1,
        )
        self.assertTrue(result.abstained)
        self.assertIsNone(result.selected_case_id)


if __name__ == "__main__":
    unittest.main()
