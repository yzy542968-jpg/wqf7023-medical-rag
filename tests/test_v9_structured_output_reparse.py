from scripts.audit_v9_structured_output_reparse import (
    balanced_json_objects,
    reparse_v9_output,
)


def test_balanced_extraction_handles_braces_inside_strings() -> None:
    text = 'prefix {"answer":"no { acute } finding","uncertainty":"low"} suffix'
    assert balanced_json_objects(text) == [
        '{"answer":"no { acute } finding","uncertainty":"low"}'
    ]


def test_reparse_skips_invalid_object_and_uses_valid_object() -> None:
    text = 'metadata {"x": 1} ```json {"answer":"Clear lungs","uncertainty":"low",}```'
    parsed, repair = reparse_v9_output(text, allowed_case_ids=[])
    assert parsed is not None
    assert parsed["answer"] == "Clear lungs"
    assert repair == "trailing_comma"


def test_reparse_does_not_fabricate_truncated_json() -> None:
    parsed, repair = reparse_v9_output(
        '{"answer":"Clear lungs","uncertainty":"low"', allowed_case_ids=[]
    )
    assert parsed is None
    assert repair == "unrecoverable"
