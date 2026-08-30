from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.multimodal.v9_generation import (  # noqa: E402
    MedGemmaImageGenerator,
    select_primary_image,
)
from medical_rag.qa.medgemma_contract import (  # noqa: E402
    build_compact_qa_prompt,
    parse_option_indices_with_wrapper_repair,
)
from medical_rag.qa.radrestruct import iter_radrestruct_cases  # noqa: E402
from medical_rag.similar_case.radgraph_adapter import (  # noqa: E402
    read_radgraph_case_records,
)
from medical_rag.similar_case.v11_evidence import (  # noqa: E402
    select_hierarchical_evidence,
)
from medical_rag.similar_case.v11_question_planner import plan_question  # noqa: E402
from run_final_qa_medgemma_contract_pilot import (  # noqa: E402
    _append_rows,
    _existing_rows,
    _read_jsonl,
    _select_rows,
    _summarize,
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _embedding_map(path: Path) -> tuple[dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as payload:
        case_ids = [str(value) for value in payload["case_ids"]]
        embeddings = np.asarray(payload["case_image_embeddings"], dtype=np.float32)
        signature = str(payload["signature"].item())
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(embeddings).all():
        raise ValueError("Image embedding cache is invalid")
    embeddings /= norms
    return dict(zip(case_ids, embeddings, strict=True)), signature


def _random_case_id(
    *,
    target_case_id: str,
    target_cluster: str,
    bank_case_ids: list[str],
    bank_clusters: dict[str, str],
    seed: int,
) -> str:
    eligible = [
        case_id
        for case_id in bank_case_ids
        if bank_clusters[case_id] != target_cluster and case_id != target_case_id
    ]
    if not eligible:
        raise ValueError("No eligible random historical case")
    return min(
        eligible,
        key=lambda case_id: hashlib.sha256(
            f"final-qa-random-history|{seed}|{target_case_id}|{case_id}".encode(
                "utf-8"
            )
        ).hexdigest(),
    )


def _whole_report_evidence(case: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    rendered: list[str] = []
    records: list[dict[str, Any]] = []
    for section in ("findings", "impression"):
        text = " ".join(str(case.get(section) or "").split())
        if not text:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rendered.append(f"{case['case_id']} | {section} | {text}")
        records.append(
            {
                "case_id": case["case_id"],
                "section": section,
                "unit_type": "whole_section",
                "unit_index": 0,
                "source_sha256": digest,
            }
        )
    return rendered, records


def _provenance_complete(records: list[dict[str, Any]]) -> bool:
    required = {"case_id", "section", "unit_type", "unit_index", "source_sha256"}
    return all(required <= set(record) and all(record[key] != "" for key in required) for record in records)


def _select_role_rows(
    manifest: dict[str, Any], radrestruct_root: Path, role: str
) -> list[dict[str, Any]]:
    if role not in manifest["roles"]:
        raise ValueError(f"Unknown Final-QA manifest role: {role}")
    role_ids = {str(case["case_id"]) for case in manifest["roles"][role]["cases"]}
    selected: list[dict[str, Any]] = []
    for case in iter_radrestruct_cases(radrestruct_root):
        if case.case_id not in role_ids:
            continue
        for index, question in enumerate(case.questions):
            selected.append(
                {
                    "case_id": case.case_id,
                    "source_report_id": case.source_report_id,
                    "official_split": case.official_split,
                    "question_index": index,
                    "question": question.question,
                    "options": list(question.options),
                    "gold_answers": list(question.answers),
                    "answer_type": question.answer_type,
                    "path": question.path,
                }
            )
    selected.sort(key=lambda row: (row["case_id"], row["question_index"]))
    if {row["case_id"] for row in selected} != role_ids:
        raise RuntimeError(f"Rad-ReStruct rows do not cover every {role} manifest case")
    return selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    config = _load_json(args.config)
    selection_config = _load_json(args.selection_config)
    manifest = _load_json(args.manifest)
    active_conditions = list(args.conditions or config["conditions"])
    unknown = sorted(set(active_conditions) - set(config["conditions"]))
    if unknown:
        raise ValueError(f"Unknown RAG pilot conditions: {unknown}")
    selected = (
        _select_role_rows(manifest, args.radrestruct_root, args.selection_role)
        if args.selection_role is not None
        else _select_rows(selection_config, manifest, args.radrestruct_root)
    )
    expected_questions = config.get("expected_question_count")
    if expected_questions is not None and len(selected) != int(expected_questions):
        raise RuntimeError(
            f"Expected {expected_questions} selected questions, found {len(selected)}"
        )
    raw_cases = {str(row["case_id"]): row for row in _read_jsonl(args.cases)}
    embeddings, embedding_signature = _embedding_map(args.embeddings)
    radgraph = read_radgraph_case_records(args.radgraph)
    train_cases = manifest["roles"]["train"]["cases"]
    bank_clusters = {case["case_id"]: case["cluster_id"] for case in train_cases}
    bank_case_ids = sorted(
        case_id
        for case_id in bank_clusters
        if case_id in raw_cases
        and case_id in embeddings
        and case_id in radgraph
        and radgraph[case_id].status == "ok"
        and (raw_cases[case_id].get("findings") or raw_cases[case_id].get("impression"))
    )
    bank_matrix = np.stack([embeddings[case_id] for case_id in bank_case_ids])
    target_manifest_role = args.selection_role or "calibration"
    cluster_by_target = {
        case["case_id"]: case["cluster_id"]
        for case in manifest["roles"][target_manifest_role]["cases"]
    }
    retrieval_by_target: dict[str, list[str]] = {}
    for target_case_id in sorted({row["case_id"] for row in selected}):
        scores = embeddings[target_case_id] @ bank_matrix.T
        eligible = np.asarray(
            [
                bank_clusters[case_id] != cluster_by_target[target_case_id]
                and case_id != target_case_id
                for case_id in bank_case_ids
            ],
            dtype=bool,
        )
        scores[~eligible] = -np.inf
        order = np.argsort(-scores, kind="stable")[:3]
        retrieval_by_target[target_case_id] = [bank_case_ids[index] for index in order]

    v12_retrieval_by_target: dict[str, list[str]] = {}
    v12_fallback_target_ids: list[str] = []
    if args.v12_ranking_rows is not None:
        ranking_rows = _read_jsonl(args.v12_ranking_rows)
        findings_rows = {
            str(row["case_id"]): row
            for row in ranking_rows
            if str(row["question_type"]) == "findings"
        }
        for target_case_id in sorted({row["case_id"] for row in selected}):
            if target_case_id not in findings_rows:
                v12_retrieval_by_target[target_case_id] = retrieval_by_target[target_case_id]
                v12_fallback_target_ids.append(target_case_id)
                continue
            ranked = [
                str(case_id)
                for case_id in findings_rows[target_case_id]["rankings"]["rrf_lambdamart"]
                if str(case_id) in bank_case_ids
                and bank_clusters[str(case_id)] != cluster_by_target[target_case_id]
                and str(case_id) != target_case_id
            ]
            if len(ranked) < 3:
                raise RuntimeError(f"V12 ranking has fewer than three eligible cases for {target_case_id}")
            v12_retrieval_by_target[target_case_id] = ranked[:3]

    evidence_cache: dict[tuple[str, str, int], tuple[list[str], list[dict[str, Any]]]] = {}
    for row in selected:
        target_case_id = row["case_id"]
        random_case_id = _random_case_id(
            target_case_id=target_case_id,
            target_cluster=cluster_by_target[target_case_id],
            bank_case_ids=bank_case_ids,
            bank_clusters=bank_clusters,
            seed=int(config["seed"]),
        )
        if "b3_no_history_r2" in active_conditions:
            evidence_cache[("b3_no_history_r2", target_case_id, row["question_index"])] = ([], [])
        if "b4_deterministic_random_history" in active_conditions:
            evidence_cache[("b4_deterministic_random_history", target_case_id, row["question_index"])] = _whole_report_evidence(raw_cases[random_case_id])
        top_ids = retrieval_by_target[target_case_id]
        if "b6_top1_image_neighbor_whole_report" in active_conditions:
            evidence_cache[("b6_top1_image_neighbor_whole_report", target_case_id, row["question_index"])] = _whole_report_evidence(raw_cases[top_ids[0]])
        p1_requested = (
            "p1_top3_image_neighbors_question_conditioned_evidence"
            in active_conditions
        )
        v12_requested = (
            "p1_v12_lambdamart_top3_question_conditioned_evidence"
            in active_conditions
        )
        if p1_requested:
            retrieved_cases = [raw_cases[case_id] for case_id in top_ids]
            plan = plan_question(
                row["question"], raw_cases[target_case_id].get("indication", "")
            )
            hierarchical = select_hierarchical_evidence(
                retrieved_cases,
                query=row["question"],
                facts_by_case={case_id: tuple(radgraph[case_id].facts) for case_id in top_ids},
                plan=plan,
                maximum_cases=3,
                maximum_units_per_case=2,
                maximum_total_units=6,
                maximum_characters=1200,
            )
            evidence_cache[("p1_top3_image_neighbors_question_conditioned_evidence", target_case_id, row["question_index"])] = (
                [f"{unit.provenance_id} | {unit.text}" for unit in hierarchical.units],
                hierarchical.as_records(),
            )
        if v12_retrieval_by_target and v12_requested:
            v12_top_ids = v12_retrieval_by_target[target_case_id]
            v12_cases = [raw_cases[case_id] for case_id in v12_top_ids]
            plan = plan_question(
                row["question"], raw_cases[target_case_id].get("indication", "")
            )
            v12_hierarchical = select_hierarchical_evidence(
                v12_cases,
                query=row["question"],
                facts_by_case={
                    case_id: tuple(radgraph[case_id].facts) for case_id in v12_top_ids
                },
                plan=plan,
                maximum_cases=3,
                maximum_units_per_case=2,
                maximum_total_units=6,
                maximum_characters=1200,
            )
            evidence_cache[(
                "p1_v12_lambdamart_top3_question_conditioned_evidence",
                target_case_id,
                row["question_index"],
            )] = (
                [f"{unit.provenance_id} | {unit.text}" for unit in v12_hierarchical.units],
                v12_hierarchical.as_records(),
            )

    existing = _existing_rows(args.rows_output)
    generator = MedGemmaImageGenerator(cache_dir=args.cache_dir, local_files_only=True)
    arm = "base"
    if args.adapter_dir is not None:
        from peft import PeftModel

        adapter_config = args.adapter_dir / "adapter_config.json"
        if not adapter_config.is_file():
            raise FileNotFoundError(adapter_config)
        generator.model = PeftModel.from_pretrained(
            generator.model,
            str(args.adapter_dir),
            is_trainable=False,
        )
        generator.model.eval()
        arm = "qlora"
    def result_key(condition: str, row: dict[str, Any]) -> str:
        base_key = f"{condition}|{row['case_id']}|{row['question_index']}"
        return base_key if arm == "base" else f"{arm}|{base_key}"

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for condition in active_conditions:
        pending = [
            row
            for row in selected
            if result_key(condition, row) not in existing
        ]
        for offset in range(0, len(pending), int(config["generation"]["batch_size"])):
            batch = pending[offset : offset + int(config["generation"]["batch_size"])]
            prompts: list[str] = []
            image_paths: list[Path] = []
            batch_evidence: list[tuple[list[str], list[dict[str, Any]]]] = []
            for row in batch:
                evidence = evidence_cache[(condition, row["case_id"], row["question_index"])]
                batch_evidence.append(evidence)
                target = raw_cases[row["case_id"]]
                prompts.append(
                    build_compact_qa_prompt(
                        question=row["question"],
                        options=row["options"],
                        indication=target.get("indication"),
                        image_available=True,
                        historical_evidence=evidence[0],
                    )
                )
                image_paths.append(select_primary_image(target, args.image_root))
            generated = generator.generate_batch(
                prompts,
                image_paths,
                max_new_tokens=int(config["generation"]["max_new_tokens"]),
                stop_token=config["generation"]["stop_token"],
            )
            completed: list[dict[str, Any]] = []
            for row, output, evidence in zip(batch, generated, batch_evidence, strict=True):
                parsed = parse_option_indices_with_wrapper_repair(
                    output["answer"],
                    option_count=len(row["options"]),
                    answer_type=row["answer_type"],
                )
                gold_indices = [
                    index
                    for index, option in enumerate(row["options"])
                    if option in set(row["gold_answers"])
                ]
                run_key = result_key(condition, row)
                record = {
                    "run_key": run_key,
                    "model_arm": arm,
                    "condition": condition,
                    "case_id": row["case_id"],
                    "question_index": row["question_index"],
                    "answer_type": row["answer_type"],
                    "path": row["path"],
                    "option_count": len(row["options"]),
                    "gold_indices": gold_indices,
                    "predicted_indices": parsed["indices"],
                    "contract_valid": parsed["contract_valid"],
                    "repairs": parsed["repairs"],
                    "raw_output": parsed["raw_output"],
                    "input_tokens": int(output["input_tokens"]),
                    "output_tokens": int(output["output_tokens"]),
                    "evidence_unit_count": len(evidence[1]),
                    "evidence_case_ids": sorted({record["case_id"] for record in evidence[1]}),
                    "provenance_complete": _provenance_complete(evidence[1]),
                }
                completed.append(record)
                existing[run_key] = record
            _append_rows(args.rows_output, completed)

    elapsed = time.perf_counter() - started
    expected_keys = {
        result_key(condition, row)
        for condition in active_conditions
        for row in selected
    }
    rows = [existing[key] for key in sorted(expected_keys)]
    if len(rows) != len(selected) * len(active_conditions):
        raise RuntimeError("RAG pilot rows are incomplete")
    conditions = {
        condition: {
            **_summarize([row for row in rows if row["condition"] == condition]),
            "mean_evidence_units": float(
                np.mean([row["evidence_unit_count"] for row in rows if row["condition"] == condition])
            ),
            "provenance_complete_rate": float(
                np.mean([row["provenance_complete"] for row in rows if row["condition"] == condition])
            ),
        }
        for condition in active_conditions
    }
    baseline = {
        (row["case_id"], row["question_index"]): set(row["predicted_indices"]) == set(row["gold_indices"])
        for row in rows
        if row["condition"] == "b3_no_history_r2"
    }
    for condition in active_conditions:
        if condition == "b3_no_history_r2":
            conditions[condition]["negative_transfer_from_b3"] = 0.0
            continue
        selected_rows = [row for row in rows if row["condition"] == condition]
        negative = sum(
            baseline[(row["case_id"], row["question_index"])]
            and set(row["predicted_indices"]) != set(row["gold_indices"])
            for row in selected_rows
        )
        eligible = sum(baseline.values())
        conditions[condition]["negative_transfer_from_b3"] = negative / eligible if eligible else None
        conditions[condition]["negative_transfer_count"] = negative
        conditions[condition]["baseline_correct_denominator"] = eligible
    summary = {
        "study": config["study"],
        "status": {
            "validation": "full_validation_generation_complete_no_test_access",
            "test": "full_test_generation_complete_frozen_configuration",
        }.get(
            str(config.get("role", "")).lower(),
            "calibration_rag_pilot_complete_no_validation_no_test",
        ),
        "model_arm": arm,
        "adapter_dir": str(args.adapter_dir.resolve()) if args.adapter_dir is not None else None,
        "config": str(args.config.resolve().relative_to(ROOT)),
        "embedding_signature": embedding_signature,
        "selected_row_count": len(selected),
        "historical_bank_case_count": len(bank_case_ids),
        "v12_ranking_coverage": {
            "requested": args.v12_ranking_rows is not None,
            "target_count": len(v12_retrieval_by_target),
            "image_only_fallback_target_count": len(v12_fallback_target_ids),
            "image_only_fallback_target_ids_sha256": hashlib.sha256(
                "\n".join(sorted(v12_fallback_target_ids)).encode("utf-8")
            ).hexdigest(),
        },
        "conditions": conditions,
        "elapsed_seconds_this_invocation": elapsed,
        "peak_vram_mb_this_invocation": torch.cuda.max_memory_allocated() / 1024**2,
        "rows_output_local_only": str(args.rows_output.resolve().relative_to(ROOT)),
        "boundary": config["boundary"],
    }
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/final_qa_report_text_rag_pilot.json")
    parser.add_argument("--selection-config", type=Path, default=ROOT / "config/final_qa_medgemma_contract_pilot_r1.json")
    parser.add_argument(
        "--selection-role",
        choices=("calibration", "validation", "test"),
        default=None,
    )
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json")
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/processed/v10_medsiglip_embeddings.npz")
    parser.add_argument("--radgraph", type=Path, default=ROOT / "data/processed/v9_radgraph_modern_xl.jsonl")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/openi_official_images")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".hf_cache")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--v12-ranking-rows", type=Path, default=None)
    parser.add_argument("--rows-output", type=Path, default=ROOT / "experiments/final_qa_development/report_text_rag_pilot_rows.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "experiments/final_qa_development/report_text_rag_pilot_summary.json")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
