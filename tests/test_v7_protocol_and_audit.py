import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_v7_config_keeps_confirmation_uninstantiated() -> None:
    config = read_json("config/v7_adaptive_fusion_development.json")
    assert config["cohort_generation"]["case_ids_instantiated"] is False
    assert config["execution"]["confirmation_case_ids_before_confirmation_protocol"] is False
    assert config["training"]["objective"] == "pairwise_logistic_ranking"
    assert config["training"]["complexity_rule"].endswith("select_linear_else_select_mlp")


def test_v7_prior_use_audit_and_development_manifest() -> None:
    audit = read_json("data/splits/v7/v7_prior_use_audit.json")
    manifest = read_json("data/splits/v7/v7_development_manifest.json")

    assert audit["status"] == "development_manifest_instantiated_confirmation_ids_not_generated"
    assert audit["v6_frame_recomputed"]["expected_counts_match"] is True
    assert audit["v7_frame"]["expected_counts_match"] is True
    assert audit["v7_frame"]["eligible_case_count"] == 1239
    assert audit["v7_frame"]["stratifiable_case_count"] == 1222
    assert audit["v7_frame"]["report_indexed_normal_count"] == 873
    assert audit["v7_frame"]["report_indexed_abnormal_count"] == 349
    assert audit["v7_frame"]["report_index_indeterminate_count"] == 17

    overlap = audit["overlap_checks"]
    assert overlap["development_with_formal_prior_use_count"] == 0
    assert overlap["development_with_post_development_confirmation_frame_count"] == 0
    assert overlap["development_with_post_development_confirmation_stratifiable_count"] == 0
    assert overlap["development_blocks_pairwise_overlap_count"] == 0
    assert overlap["all_required_disjointness_checks_zero"] is True

    assert manifest["status"] == "development_blocks_instantiated_confirmation_ids_not_generated"
    assert manifest["confirmation_case_ids_instantiated"] is False
    assert manifest["development_case_count"] == 720
    assert set(manifest["blocks"]) == {"train_a", "train_b", "validation"}
    assert "confirmation" not in manifest["blocks"]
    for block in manifest["blocks"].values():
        assert block["case_count"] == 240
        assert block["report_indexed_normal"] == 172
        assert block["report_indexed_abnormal"] == 68
        assert len(block["target_case_ids"]) == 120
        assert len(block["distractor_case_ids"]) == 120


def test_v7_confirmation_config_is_frozen_before_case_instantiation() -> None:
    config = read_json("config/v7_confirmation.json")
    assert config["status"] == "confirmation_protocol_frozen_before_confirmation_case_instantiation"
    assert config["cohort_generation"]["case_ids_instantiated"] is False
    assert config["execution"]["confirmation_case_ids_generated"] is False
    assert config["adaptive_model"]["model_type"] == "linear_sigmoid"
    assert config["adaptive_model"]["epochs"] == 6
    assert config["retrieval"]["global_alpha_star"] == 0.52
    assert config["statistics"]["h2_success"] == "plus_one_p_le_0.05"
