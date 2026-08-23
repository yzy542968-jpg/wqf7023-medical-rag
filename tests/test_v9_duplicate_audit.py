from pathlib import Path

from PIL import Image

from scripts.audit_v9_cross_split_duplicates import dhash64, normalized_report


def test_normalized_report_is_case_and_whitespace_stable() -> None:
    left = {"findings": "  Clear   LUNGS ", "impression": "No acute disease."}
    right = {"findings": "clear lungs", "impression": "no acute disease."}
    assert normalized_report(left) == normalized_report(right)


def test_dhash_is_identical_for_identical_pixels(tmp_path: Path) -> None:
    image = Image.new("L", (12, 12), color=128)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    image.save(first)
    image.save(second)
    assert dhash64(first) == dhash64(second)
