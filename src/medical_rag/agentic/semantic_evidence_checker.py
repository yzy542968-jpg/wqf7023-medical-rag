from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from medical_rag.agentic.evidence_checker import score_sentence_pair, split_sentences


DEFAULT_NLI_MODEL = "cnut1648/biolinkbert-mednli"


class NLIPredictor(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]: ...


class MedicalNLIPredictor:
    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional environment
            raise RuntimeError("Medical NLI requires torch and transformers") from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, local_files_only=local_files_only
        ).to(self.device)
        self.model.eval()
        self.id2label = {
            int(index): str(label).lower()
            for index, label in self.model.config.id2label.items()
        }
        required = {"entailment", "neutral", "contradiction"}
        if not required.issubset(set(self.id2label.values())):
            raise ValueError(f"NLI model labels must include {sorted(required)}")

    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        predictions: list[dict[str, float]] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = self.tokenizer(
                [premise for premise, _ in batch],
                [hypothesis for _, hypothesis in batch],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.inference_mode():
                probabilities = self.torch.softmax(self.model(**encoded).logits, dim=-1)
            for probability_row in probabilities.detach().cpu().tolist():
                mapped = {
                    self.id2label[index]: float(value)
                    for index, value in enumerate(probability_row)
                }
                predictions.append(
                    {
                        "entailment": mapped["entailment"],
                        "neutral": mapped["neutral"],
                        "contradiction": mapped["contradiction"],
                    }
                )
        return predictions


@dataclass
class SemanticEvidenceSentenceCheck:
    sentence: str
    matched_evidence: str
    lexical_score: float
    entailment_probability: float
    neutral_probability: float
    contradiction_probability: float
    combined_support_score: float
    negation_consistent: bool
    supported: bool
    decision_reason: str


@dataclass
class SemanticEvidenceCheckResult:
    supported_sentences: list[str]
    unsupported_sentences: list[str]
    sentence_checks: list[SemanticEvidenceSentenceCheck]
    support_rate: float
    revised_answer: str
    abstained: bool


def _decision_reason(
    *,
    supported: bool,
    negation_consistent: bool,
    contradiction_probability: float,
    contradiction_threshold: float,
    entailment_probability: float,
    entailment_threshold: float,
) -> str:
    if not negation_consistent:
        return "rule_polarity_conflict"
    if contradiction_probability >= contradiction_threshold:
        return "nli_contradiction"
    if supported and entailment_probability >= entailment_threshold:
        return "nli_entailment"
    if supported:
        return "combined_support"
    return "insufficient_support"


def check_semantic_evidence_support(
    answer: str,
    evidence_text: str,
    predictor: NLIPredictor,
    *,
    min_combined_support: float = 0.55,
    entailment_threshold: float = 0.75,
    contradiction_threshold: float = 0.50,
    lexical_weight: float = 0.35,
    max_candidates: int = 6,
    polarity_conflict_lexical_threshold: float = 0.45,
) -> SemanticEvidenceCheckResult:
    if not 0.0 <= lexical_weight <= 1.0:
        raise ValueError("lexical_weight must be between 0 and 1")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if not 0.0 <= polarity_conflict_lexical_threshold <= 1.0:
        raise ValueError("polarity conflict lexical threshold must be between 0 and 1")

    answer_sentences = split_sentences(answer)
    evidence_sentences = split_sentences(evidence_text)
    if not answer_sentences or not evidence_sentences:
        return SemanticEvidenceCheckResult(
            supported_sentences=[],
            unsupported_sentences=answer_sentences,
            sentence_checks=[],
            support_rate=0.0,
            revised_answer="The retrieved report evidence is insufficient to answer this question.",
            abstained=True,
        )

    candidate_groups: list[list[tuple[str, float, bool]]] = []
    flattened_pairs: list[tuple[str, str]] = []
    for answer_sentence in answer_sentences:
        candidates = []
        for evidence_sentence in evidence_sentences:
            lexical_score, negation_consistent = score_sentence_pair(
                answer_sentence, evidence_sentence
            )
            candidates.append((evidence_sentence, lexical_score, negation_consistent))
        candidates.sort(key=lambda value: value[1], reverse=True)
        selected_candidates = candidates[: min(max_candidates, len(candidates))]
        candidate_groups.append(selected_candidates)
        flattened_pairs.extend(
            (evidence_sentence, answer_sentence)
            for evidence_sentence, _, _ in selected_candidates
        )

    nli_predictions = predictor.predict(flattened_pairs)
    if len(nli_predictions) != len(flattened_pairs):
        raise ValueError("NLI predictor returned an unexpected number of predictions")

    sentence_checks: list[SemanticEvidenceSentenceCheck] = []
    supported_sentences: list[str] = []
    unsupported_sentences: list[str] = []
    prediction_index = 0
    semantic_weight = 1.0 - lexical_weight

    for answer_sentence, candidates in zip(answer_sentences, candidate_groups, strict=True):
        scored_candidates = []
        for evidence_sentence, lexical_score, negation_consistent in candidates:
            nli = nli_predictions[prediction_index]
            prediction_index += 1
            relation_strength = max(nli["entailment"], nli["contradiction"])
            alignment_score = 0.30 * lexical_score + 0.70 * relation_strength
            scored_candidates.append(
                (
                    alignment_score,
                    evidence_sentence,
                    lexical_score,
                    negation_consistent,
                    nli,
                )
            )

        polarity_conflicts = [
            value
            for value in scored_candidates
            if not value[3] and value[2] >= polarity_conflict_lexical_threshold
        ]
        if polarity_conflicts:
            # A high-overlap polarity conflict is more reliable than NLI on
            # short fragments and cannot be overridden by an unrelated false
            # entailment.
            _, matched_evidence, lexical_score, negation_consistent, nli = max(
                polarity_conflicts, key=lambda value: value[2]
            )
        else:
            _, matched_evidence, lexical_score, negation_consistent, nli = max(
                scored_candidates, key=lambda value: value[0]
            )
        combined_support = lexical_weight * lexical_score + semantic_weight * nli["entailment"]
        supported = (
            negation_consistent
            and nli["contradiction"] < contradiction_threshold
            and (
                combined_support >= min_combined_support
                or nli["entailment"] >= entailment_threshold
            )
        )
        reason = _decision_reason(
            supported=supported,
            negation_consistent=negation_consistent,
            contradiction_probability=nli["contradiction"],
            contradiction_threshold=contradiction_threshold,
            entailment_probability=nli["entailment"],
            entailment_threshold=entailment_threshold,
        )
        check = SemanticEvidenceSentenceCheck(
            sentence=answer_sentence,
            matched_evidence=matched_evidence,
            lexical_score=lexical_score,
            entailment_probability=nli["entailment"],
            neutral_probability=nli["neutral"],
            contradiction_probability=nli["contradiction"],
            combined_support_score=combined_support,
            negation_consistent=negation_consistent,
            supported=supported,
            decision_reason=reason,
        )
        sentence_checks.append(check)
        if supported:
            supported_sentences.append(answer_sentence)
        else:
            unsupported_sentences.append(answer_sentence)

    support_rate = len(supported_sentences) / len(sentence_checks)
    abstained = not supported_sentences
    revised_answer = (
        " ".join(supported_sentences)
        if supported_sentences
        else "The retrieved report evidence is insufficient to answer this question."
    )
    return SemanticEvidenceCheckResult(
        supported_sentences=supported_sentences,
        unsupported_sentences=unsupported_sentences,
        sentence_checks=sentence_checks,
        support_rate=support_rate,
        revised_answer=revised_answer,
        abstained=abstained,
    )
