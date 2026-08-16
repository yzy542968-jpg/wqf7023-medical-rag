from __future__ import annotations

import unittest

from medical_rag.agentic.semantic_evidence_checker import check_semantic_evidence_support


class FakeNLIPredictor:
    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        outputs = []
        for premise, hypothesis in pairs:
            combined = f"{premise.lower()} || {hypothesis.lower()}"
            if "cardiomegaly" in combined and "cardiac silhouette is enlarged" in combined:
                outputs.append({"entailment": 0.92, "neutral": 0.06, "contradiction": 0.02})
            elif "no focal consolidation" in premise.lower() and "focal consolidation is present" in hypothesis.lower():
                outputs.append({"entailment": 0.01, "neutral": 0.02, "contradiction": 0.97})
            elif "unrelated" in premise.lower():
                outputs.append({"entailment": 0.05, "neutral": 0.90, "contradiction": 0.05})
            else:
                outputs.append({"entailment": 0.85, "neutral": 0.10, "contradiction": 0.05})
        return outputs


class MisleadingNLIPredictor:
    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        outputs = []
        for premise, _ in pairs:
            if "unrelated" in premise.lower():
                outputs.append({"entailment": 0.999, "neutral": 0.0005, "contradiction": 0.0005})
            else:
                outputs.append({"entailment": 0.01, "neutral": 0.98, "contradiction": 0.01})
        return outputs


class SemanticEvidenceCheckerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = FakeNLIPredictor()

    def test_semantic_entailment_supports_low_lexical_overlap(self) -> None:
        result = check_semantic_evidence_support(
            "The cardiac silhouette is enlarged.",
            "Mild cardiomegaly.",
            self.predictor,
        )
        self.assertFalse(result.abstained)
        self.assertTrue(result.sentence_checks[0].supported)
        self.assertEqual(result.sentence_checks[0].decision_reason, "nli_entailment")

    def test_nli_contradiction_rejects_lexically_similar_claim(self) -> None:
        result = check_semantic_evidence_support(
            "Focal consolidation is present.",
            "No focal consolidation.",
            self.predictor,
        )
        self.assertTrue(result.abstained)
        self.assertFalse(result.sentence_checks[0].supported)

    def test_rule_polarity_conflict_overrides_entailment(self) -> None:
        result = check_semantic_evidence_support(
            "There is pleural effusion.",
            "There is no pleural effusion.",
            self.predictor,
        )
        self.assertTrue(result.abstained)
        self.assertEqual(result.sentence_checks[0].decision_reason, "rule_polarity_conflict")

    def test_high_overlap_polarity_guard_beats_unrelated_false_entailment(self) -> None:
        result = check_semantic_evidence_support(
            "Acute osseous abnormality.",
            "This is unrelated. No acute osseous abnormality.",
            MisleadingNLIPredictor(),
        )
        self.assertTrue(result.abstained)
        self.assertEqual(result.sentence_checks[0].matched_evidence, "No acute osseous abnormality.")
        self.assertEqual(result.sentence_checks[0].decision_reason, "rule_polarity_conflict")

    def test_neutral_evidence_is_not_supported(self) -> None:
        result = check_semantic_evidence_support(
            "A right pleural effusion is present.",
            "This is unrelated evidence.",
            self.predictor,
        )
        self.assertTrue(result.abstained)

    def test_supported_sentences_are_kept_and_unsupported_removed(self) -> None:
        result = check_semantic_evidence_support(
            "Mild cardiomegaly. Focal consolidation is present.",
            "The cardiac silhouette is enlarged. No focal consolidation.",
            self.predictor,
        )
        self.assertIn("Mild cardiomegaly.", result.revised_answer)
        self.assertNotIn("Focal consolidation is present.", result.revised_answer)


if __name__ == "__main__":
    unittest.main()
