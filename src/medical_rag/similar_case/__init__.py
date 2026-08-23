"""Leakage-aware utilities for other-patient similar-case retrieval."""

from medical_rag.similar_case.bank import CandidateBankAudit, build_candidate_bank
from medical_rag.similar_case.chexpert_plus_adapter import read_chexpert_plus_cases
from medical_rag.similar_case.relevance import build_query_qrels, report_relevance_gain
from medical_rag.similar_case.radgraph_adapter import read_radgraph_facts_by_text
from medical_rag.similar_case.schema import PairedCase
from medical_rag.similar_case.text_baseline import SimilarCaseBM25Retriever

__all__ = [
    "CandidateBankAudit",
    "PairedCase",
    "SimilarCaseBM25Retriever",
    "build_candidate_bank",
    "build_query_qrels",
    "read_chexpert_plus_cases",
    "read_radgraph_facts_by_text",
    "report_relevance_gain",
]
