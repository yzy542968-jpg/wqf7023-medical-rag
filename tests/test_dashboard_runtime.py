from pathlib import Path

from medical_rag.dashboard.demo_generation import extractive_demo_answer
from medical_rag.dashboard.runtime import resolve_dashboard_runtime


def test_fresh_clone_uses_tracked_demo_cases(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MEDICAL_RAG_DEMO_MODE", raising=False)
    demo_path = tmp_path / "data" / "processed" / "sample_cases.jsonl"
    demo_path.parent.mkdir(parents=True)
    demo_path.write_text('{"case_id":"demo"}\n', encoding="utf-8")

    runtime = resolve_dashboard_runtime(tmp_path)

    assert runtime.is_demo
    assert runtime.cases_path == demo_path
    assert not runtime.dense_retrieval_available


def test_full_runtime_can_degrade_without_dense_index(tmp_path: Path) -> None:
    cases_path = tmp_path / "data" / "processed" / "openi_cases.jsonl"
    cases_path.parent.mkdir(parents=True)
    cases_path.write_text('{"case_id":"full"}\n', encoding="utf-8")

    runtime = resolve_dashboard_runtime(tmp_path)

    assert not runtime.is_demo
    assert not runtime.dense_retrieval_available


def test_demo_mode_can_be_forced_with_full_data_present(tmp_path: Path, monkeypatch) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "openi_cases.jsonl").write_text('{"case_id":"full"}\n', encoding="utf-8")
    (processed / "sample_cases.jsonl").write_text('{"case_id":"demo"}\n', encoding="utf-8")
    monkeypatch.setenv("MEDICAL_RAG_DEMO_MODE", "1")

    assert resolve_dashboard_runtime(tmp_path).is_demo


def test_extract_demo_answer_prefers_requested_report_field() -> None:
    prompt = """Question:\nWhat is the impression?\n\nSelected radiology case:\nCase ID: CXR2\nFindings: Right lower lobe opacity.\nImpression: Right lower lobe pneumonia.\n"""

    assert extractive_demo_answer(prompt) == "Right lower lobe pneumonia."
