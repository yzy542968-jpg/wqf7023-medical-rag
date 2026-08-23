from pathlib import Path

from PIL import Image

from medical_rag.multimodal.v9_generation import (
    build_v9_qa_prompt,
    parse_v9_output,
    select_primary_image,
)


def test_primary_image_prefers_frontal(tmp_path: Path) -> None:
    Image.new("RGB", (2, 2)).save(tmp_path / "CXRlateral.png")
    Image.new("RGB", (2, 2)).save(tmp_path / "CXRfrontal.png")
    case = {
        "case_id": "CXR1",
        "images": [
            {"filename": "lateral.png", "projection": "Lateral"},
            {"filename": "frontal.png", "projection": "Frontal"},
        ],
    }
    assert select_primary_image(case, tmp_path).name == "CXRfrontal.png"


def test_prompt_marks_historical_reports_as_analogies() -> None:
    prompt = build_v9_qa_prompt(
        "What are the findings?",
        "Cough",
        [{"case_id": "CXR2", "findings": "Opacity.", "impression": "Pneumonia."}],
    )
    assert "other-patient analogies" in prompt
    assert "CXR2" in prompt
    assert "Cough" in prompt


def test_parser_filters_unknown_citations() -> None:
    result = parse_v9_output(
        '{"answer":"Clear lungs.","target_image_findings":["Clear lungs"],'
        '"supporting_case_ids":["CXR2","OTHER"],"historical_support":"Similar",'
        '"uncertainty":"low","abstain":false}',
        ["CXR2"],
    )
    assert result["structured_output_valid"] is True
    assert result["supporting_case_ids"] == ["CXR2"]
    assert result["answer"] == "Clear lungs."


def test_parser_retains_unstructured_answer() -> None:
    result = parse_v9_output("No focal airspace disease.", [])
    assert result["structured_output_valid"] is False
    assert result["answer"] == "No focal airspace disease."
