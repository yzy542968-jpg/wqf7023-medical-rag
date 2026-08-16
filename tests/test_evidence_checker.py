from __future__ import annotations

import unittest

from medical_rag.agentic.evidence_checker import check_evidence_support


class EvidenceCheckerTests(unittest.TestCase):
    def test_positive_claim_matches_positive_evidence(self) -> None:
        result = check_evidence_support(
            "There is right basilar atelectasis.",
            "Findings: There is right basilar atelectasis.",
        )
        self.assertEqual(result.support_rate, 1.0)
        self.assertFalse(result.abstained)

    def test_negative_claim_matches_negative_evidence(self) -> None:
        result = check_evidence_support(
            "There is no pneumothorax.",
            "Findings: There is no evidence of pneumothorax.",
            min_sentence_support=0.5,
        )
        self.assertEqual(result.support_rate, 1.0)
        self.assertTrue(result.sentence_checks[0].negation_consistent)

    def test_positive_claim_is_rejected_by_negative_evidence(self) -> None:
        result = check_evidence_support(
            "There is pneumothorax.",
            "Findings: There is no pneumothorax.",
            min_sentence_support=0.5,
        )
        self.assertEqual(result.support_rate, 0.0)
        self.assertTrue(result.abstained)
        self.assertFalse(result.sentence_checks[0].negation_consistent)

    def test_negative_claim_is_rejected_by_positive_evidence(self) -> None:
        result = check_evidence_support(
            "There is no pleural effusion.",
            "Findings: A small left pleural effusion is present.",
            min_sentence_support=0.5,
        )
        self.assertEqual(result.support_rate, 0.0)
        self.assertTrue(result.abstained)

    def test_best_evidence_sentence_prevents_cross_sentence_token_union(self) -> None:
        result = check_evidence_support(
            "There is a right upper lobe opacity.",
            "The right lung is clear. There is a left lower lobe opacity.",
            min_sentence_support=0.8,
        )
        self.assertEqual(result.support_rate, 0.0)

    def test_unsupported_sentence_is_removed_during_revision(self) -> None:
        result = check_evidence_support(
            "There is right basilar atelectasis. There is a pleural effusion.",
            "Findings: There is right basilar atelectasis. There is no pleural effusion.",
            min_sentence_support=0.6,
        )
        self.assertEqual(result.supported_sentences, ["There is right basilar atelectasis."])
        self.assertEqual(result.unsupported_sentences, ["There is a pleural effusion."])
        self.assertEqual(result.revised_answer, "There is right basilar atelectasis.")

    def test_plural_finding_matches_singular_claim(self) -> None:
        result = check_evidence_support(
            "There is opacity.",
            "There are bilateral interstitial opacities.",
            min_sentence_support=0.4,
        )
        self.assertEqual(result.support_rate, 1.0)

    def test_normal_and_not_abnormal_are_polarity_compatible(self) -> None:
        result = check_evidence_support(
            "The heart size is within normal limits, so it is not abnormal.",
            "The cardiac silhouette is within normal limits.",
            min_sentence_support=0.4,
        )
        self.assertTrue(result.sentence_checks[0].negation_consistent)


if __name__ == "__main__":
    unittest.main()
