from __future__ import annotations

import unittest

from medical_rag.evaluation.splits import build_grouped_case_split, filter_questions_for_split


class GroupedCaseSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.questions = [
            {
                "qid": f"case{case_index}_q{question_index}",
                "case_id": f"case{case_index}",
                "question_type": f"type{question_index}",
            }
            for case_index in range(10)
            for question_index in range(3)
        ]

    def test_split_is_grouped_complete_and_reproducible(self) -> None:
        first = build_grouped_case_split(self.questions, development_fraction=0.2, seed=7)
        second = build_grouped_case_split(self.questions, development_fraction=0.2, seed=7)

        self.assertEqual(first, second)
        development_cases = set(first["development"]["case_ids"])
        test_cases = set(first["test"]["case_ids"])
        self.assertFalse(development_cases.intersection(test_cases))
        self.assertEqual(first["development"]["case_count"], 2)
        self.assertEqual(first["development"]["question_count"], 6)
        self.assertEqual(first["test"]["question_count"], 24)

    def test_filter_preserves_original_question_order(self) -> None:
        split = build_grouped_case_split(self.questions, development_fraction=0.2, seed=7)
        selected = filter_questions_for_split(self.questions, split, "development")
        selected_qids = [item["qid"] for item in selected]
        expected_qids = [
            item["qid"]
            for item in self.questions
            if item["qid"] in set(split["development"]["qids"])
        ]
        self.assertEqual(selected_qids, expected_qids)

    def test_duplicate_question_ids_are_rejected(self) -> None:
        duplicated = self.questions + [self.questions[0].copy()]
        with self.assertRaises(ValueError):
            build_grouped_case_split(duplicated)


if __name__ == "__main__":
    unittest.main()
