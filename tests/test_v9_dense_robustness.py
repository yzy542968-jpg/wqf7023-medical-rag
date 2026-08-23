from scripts.run_v9_supplemental_dense_robustness import (
    aggregate_robustness,
    instructed_query,
    local_model_snapshot,
    top_k_jaccard,
)


def test_instruction_format_is_frozen() -> None:
    assert instructed_query("retrieve reports", "indication\nquestion") == (
        "Instruct: retrieve reports\nQuery:indication\nquestion"
    )


def test_top_k_jaccard() -> None:
    assert top_k_jaccard(["a", "b"], ["b", "c"]) == 1 / 3


def test_local_snapshot_requires_pinned_files(tmp_path) -> None:
    snapshot = tmp_path / "models--owner--model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "tokenizer.json", "modules.json"):
        (snapshot / name).write_text("{}", encoding="utf-8")
    assert local_model_snapshot("owner/model", "revision", tmp_path) == snapshot


def test_aggregate_robustness_uses_canonical_variant() -> None:
    rows = [
        {
            "case_id": "case",
            "question_type": "findings",
            "variant_index": index,
            "question": f"q{index}",
            "ndcg@10": value,
            "top1_case_id": top1,
            "top10_case_ids": top10,
        }
        for index, value, top1, top10 in (
            (0, 0.3, "a", "a;b"),
            (1, 0.2, "a", "a;c"),
            (2, 0.4, "d", "d;b"),
        )
    ]
    result = aggregate_robustness(rows)
    assert result["canonical_ndcg@10"] == 0.3
    assert result["top1_consistency_with_canonical"] == 0.5
    assert result["case_role_count"] == 1
