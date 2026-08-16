from __future__ import annotations

from medical_rag.evaluation.case_scoped_benchmark import (
    benchmark_summary,
    build_case_chunks,
    build_case_scoped_benchmark,
    content_fingerprint,
    expected_section,
)
from medical_rag.retrieval.scoped_chunk_retriever import ScopedBM25ChunkRetriever


def sample_case(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "indication": "Cough.",
        "comparison": "None.",
        "findings": "There is a focal opacity. No pleural effusion is present.",
        "impression": "Focal pulmonary opacity.",
        "problems": "Opacity",
        "images": [{"filename": f"{case_id}.png", "projection": "Frontal"}],
    }


def test_chunks_have_stable_case_section_ids() -> None:
    chunks = build_case_chunks(sample_case("CXR1"))
    assert chunks[0]["chunk_id"] == "CXR1::indication::001"
    assert {row["section"] for row in chunks} == {"indication", "comparison", "findings", "impression"}


def test_benchmark_excludes_prior_cases_and_has_three_disjoint_splits() -> None:
    cases = [sample_case(f"CXR{i}") for i in range(11)]
    payload = build_case_scoped_benchmark(cases, {"CXR0"}, max_cases=10, seed=7)
    parts = [set(payload["split"][name]["case_ids"]) for name in ("development", "calibration", "test")]
    assert not parts[0] & parts[1]
    assert not parts[0] & parts[2]
    assert not parts[1] & parts[2]
    assert "CXR0" not in set().union(*parts)
    assert payload["question_count"] == 30
    assert benchmark_summary(payload)["case_scoped_unique_query_rate"] == 1.0


def test_scoped_retriever_never_returns_another_case() -> None:
    chunks = build_case_chunks(sample_case("CXR1")) + build_case_chunks(sample_case("CXR2"))
    retriever = ScopedBM25ChunkRetriever().fit(chunks)
    results = retriever.search("focal opacity", top_k=5, case_id="CXR2")
    assert results
    assert {row["case_id"] for row in results} == {"CXR2"}


def test_agent_route_maps_question_type_to_gold_section() -> None:
    assert expected_section("case_scoped_findings") == "findings"
    assert expected_section("case_scoped_summary") == "impression"


def test_content_fingerprint_detects_text_changes() -> None:
    chunks = build_case_chunks(sample_case("CXR1"))
    baseline = content_fingerprint([], chunks)
    changed = [dict(row) for row in chunks]
    changed[0]["text"] = "Changed indication."
    assert content_fingerprint([], changed) != baseline
