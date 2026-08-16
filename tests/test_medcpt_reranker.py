from __future__ import annotations

import unittest

from medical_rag.retrieval.medcpt_reranker import case_document, rerank_by_scores


class MedCPTRerankerTests(unittest.TestCase):
    def test_rerank_orders_candidates_by_descending_score(self) -> None:
        candidates = [{"case_id": "a"}, {"case_id": "b"}, {"case_id": "c"}]
        reranked = rerank_by_scores(candidates, [0.2, 0.9, -0.1])
        self.assertEqual([item["case_id"] for item in reranked], ["b", "a", "c"])
        self.assertEqual([item["reranked_rank"] for item in reranked], [1, 2, 3])

    def test_rerank_rejects_mismatched_score_count(self) -> None:
        with self.assertRaises(ValueError):
            rerank_by_scores([{"case_id": "a"}], [])

    def test_case_document_contains_retrieval_fields(self) -> None:
        text = case_document(
            {
                "indication": "Cough",
                "problems": "Opacity",
                "findings": "Right opacity.",
                "impression": "Possible pneumonia.",
            }
        )
        self.assertIn("Indication: Cough", text)
        self.assertIn("Findings: Right opacity.", text)


if __name__ == "__main__":
    unittest.main()
