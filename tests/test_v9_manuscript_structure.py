from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "docs" / "P2_V9_FINAL_MANUSCRIPT.md"


def test_v9_is_the_primary_manuscript_structure() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    for chapter in range(1, 6):
        assert text.count(f"# Chapter {chapter}:") == 1
    assert "## 3.1 Final Research Design and Version Boundary" in text
    assert "## 4.1 Final V9 Retrieval Confirmation" in text
    assert "## Appendix G: Frozen Preliminary Controlled-Study Methods" in text
    assert "## Appendix H: Frozen Preliminary Controlled-Study Results" in text


def test_manuscript_has_final_review_and_expected_length() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    words = text.split()
    assert 10_000 < len(words) < 30_000
    assert "accepted the tool-assisted taxonomy labels without modification" in text
    assert "researcher review remains pending" not in text
    assert "awaits researcher review" not in text


def test_manuscript_contains_supplemental_validity_boundaries() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    assert "## 4.7 Cross-Split Similarity Sensitivity" in text
    assert "## 4.8 Qrel, Dense-Baseline, and Wording Robustness" in text
    assert "## 4.9 Clinical Semantic and Structured-Output Audits" in text
    assert "does not demonstrate external generalization" in text
