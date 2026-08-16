from __future__ import annotations

import re
from dataclasses import dataclass


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "based",
    "described",
    "evidence",
    "for",
    "finding",
    "findings",
    "from",
    "has",
    "have",
    "in",
    "indicating",
    "is",
    "it",
    "noted",
    "of",
    "on",
    "or",
    "report",
    "section",
    "the",
    "there",
    "this",
    "to",
    "was",
    "were",
    "with",
    "within",
    "yes",
}

NEGATION_RE = re.compile(
    r"\b(?:no|not|without|absent|absence\s+of|negative\s+for|free\s+of|"
    r"neither|nor|cannot|can't|unable\s+to)\b",
    re.IGNORECASE,
)

NORMALITY_RE = re.compile(
    r"\b(?:normal|within\s+normal\s+limits|unremarkable|no\s+acute\s+(?:disease|abnormality))\b",
    re.IGNORECASE,
)


@dataclass
class EvidenceSentenceCheck:
    sentence: str
    support_score: float
    supported: bool
    matched_evidence: str = ""
    negation_consistent: bool = True


@dataclass
class EvidenceCheckResult:
    supported_sentences: list[str]
    unsupported_sentences: list[str]
    sentence_checks: list[EvidenceSentenceCheck]
    support_rate: float
    revised_answer: str
    abstained: bool


def _sentences(text: str) -> list[str]:
    protected = re.sub(r"\b(\d+)\.\s+", r"\1<NUMDOT> ", (text or "").strip())
    parts = re.split(r"(?<=[.!?])\s+|[\r\n]+|\s*;\s*", protected)
    return [part.replace("<NUMDOT>", ".").strip() for part in parts if part.strip()]


def _content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {_normalize_token(token) for token in tokens if token not in STOPWORDS and len(token) > 1}


def _normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _contains_negation(text: str) -> bool:
    return bool(NEGATION_RE.search(text or "") or NORMALITY_RE.search(text or ""))


def _polarity_compatible(sentence: str, evidence_sentence: str) -> bool:
    return _contains_negation(sentence) == _contains_negation(evidence_sentence)


def split_sentences(text: str) -> list[str]:
    return _sentences(text)


def score_sentence_pair(sentence: str, evidence_sentence: str) -> tuple[float, bool]:
    normalized_sentence = " ".join(sentence.lower().split())
    sentence_tokens = _content_tokens(sentence)
    if not sentence_tokens:
        return 0.0, True

    negation_consistent = _polarity_compatible(sentence, evidence_sentence)
    normalized_evidence = " ".join(evidence_sentence.lower().split())
    if normalized_sentence and normalized_sentence in normalized_evidence:
        score = 1.0
    else:
        evidence_tokens = _content_tokens(evidence_sentence)
        score = len(sentence_tokens.intersection(evidence_tokens)) / len(sentence_tokens)
    # Preserve lexical similarity even when polarity conflicts. Callers use
    # the separate boolean as a hard rejection signal; zeroing the score here
    # can make an unrelated sentence win evidence alignment.
    return score, negation_consistent


def _support_score(sentence: str, evidence_text: str) -> tuple[float, str, bool]:
    sentence_tokens = _content_tokens(sentence)
    if not sentence_tokens:
        return 0.0, "", True

    best_score = 0.0
    best_evidence = ""
    best_negation_consistent = True
    for evidence_sentence in _sentences(evidence_text):
        score, negation_consistent = score_sentence_pair(sentence, evidence_sentence)

        if score > best_score or (not best_evidence and score == best_score):
            best_score = score
            best_evidence = evidence_sentence
            best_negation_consistent = negation_consistent

    return best_score, best_evidence, best_negation_consistent


def check_evidence_support(
    answer: str,
    evidence_text: str,
    min_sentence_support: float = 0.65,
) -> EvidenceCheckResult:
    checks: list[EvidenceSentenceCheck] = []
    supported_sentences: list[str] = []
    unsupported_sentences: list[str] = []

    for sentence in _sentences(answer):
        score, matched_evidence, negation_consistent = _support_score(sentence, evidence_text)
        supported = negation_consistent and score >= min_sentence_support
        checks.append(
            EvidenceSentenceCheck(
                sentence=sentence,
                support_score=score,
                supported=supported,
                matched_evidence=matched_evidence,
                negation_consistent=negation_consistent,
            )
        )
        if supported:
            supported_sentences.append(sentence)
        else:
            unsupported_sentences.append(sentence)

    if not checks:
        return EvidenceCheckResult(
            supported_sentences=[],
            unsupported_sentences=[],
            sentence_checks=[],
            support_rate=0.0,
            revised_answer="The retrieved report evidence is insufficient to answer this question.",
            abstained=True,
        )

    support_rate = len(supported_sentences) / len(checks)
    abstained = not supported_sentences
    revised_answer = (
        " ".join(supported_sentences)
        if supported_sentences
        else "The retrieved report evidence is insufficient to answer this question."
    )

    return EvidenceCheckResult(
        supported_sentences=supported_sentences,
        unsupported_sentences=unsupported_sentences,
        sentence_checks=checks,
        support_rate=support_rate,
        revised_answer=revised_answer,
        abstained=abstained,
    )
