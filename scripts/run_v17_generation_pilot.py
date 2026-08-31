"""Run the frozen V17 matched-history MedGemma Calibration pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.multimodal.v9_generation import MedGemmaImageGenerator, select_primary_image  # noqa: E402
from medical_rag.qa.medgemma_contract import build_compact_qa_prompt, parse_option_indices_with_wrapper_repair  # noqa: E402
from medical_rag.qa.radrestruct import iter_radrestruct_cases  # noqa: E402
from medical_rag.similar_case.radgraph_adapter import read_radgraph_case_records  # noqa: E402
from medical_rag.similar_case.v11_evidence import evidence_profile, select_hierarchical_evidence  # noqa: E402
from medical_rag.similar_case.v11_question_planner import plan_question  # noqa: E402


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _selected_questions(root: Path, case_ids: set[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for case in iter_radrestruct_cases(root):
        if case.case_id not in case_ids:
            continue
        for index, question in enumerate(case.questions):
            selected.append(
                {
                    "case_id": case.case_id,
                    "question_index": index,
                    "question": question.question,
                    "options": list(question.options),
                    "gold_answers": list(question.answers),
                    "answer_type": question.answer_type,
                    "path": question.path,
                }
            )
    return sorted(selected, key=lambda row: (row["case_id"], row["question_index"]))


def _whole_report(case: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], dict[str, float]]:
    rendered: list[str] = []
    records: list[dict[str, Any]] = []
    character_count = 0
    for section in ("findings", "impression"):
        text = " ".join(str(case.get(section) or "").split())
        if text:
            case_id = str(case["case_id"])
            rendered.append(f"{case_id} | {section} | {text}")
            records.append(
                {
                    "provenance_id": f"{case_id}:{section}:whole_section:0",
                    "case_id": case_id,
                    "section": section,
                    "unit_type": "whole_section",
                    "unit_index": 0,
                    "text": text,
                    "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "score": 0.0,
                }
            )
            character_count += len(text)
    return (
        rendered,
        records,
        {
            "unit_count": float(len(records)),
            "case_count": float(bool(records)),
            "sentence_count": 0.0,
            "fact_count": 0.0,
            "character_count": float(character_count),
            "provenance_complete_rate": 1.0,
            "duplicate_text_rate": 0.0,
        },
    )


def _metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    if not values:
        raise ValueError("Cannot score an empty condition")
    tp = fp = fn = 0
    exact: list[float] = []
    valid: list[float] = []
    strata: dict[str, list[float]] = defaultdict(list)
    families: dict[str, list[float]] = defaultdict(list)
    for row in values:
        gold = set(int(value) for value in row["gold_indices"])
        predicted = set(int(value) for value in row["predicted_indices"])
        match = float(gold == predicted)
        exact.append(match)
        valid.append(float(row["contract_valid"]))
        strata[str(row["stratum"])].append(match)
        families[str(row["path"]).split("_", 1)[0] or "missing"].append(match)
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    denominator = 2 * tp + fp + fn
    stratum_accuracy = {name: float(np.mean(scores)) for name, scores in sorted(strata.items())}
    return {
        "row_count": len(values),
        "exact_answer_set_accuracy": float(np.mean(exact)),
        "option_micro_f1": 2 * tp / denominator if denominator else 0.0,
        "contract_valid_rate": float(np.mean(valid)),
        "stratum_exact_accuracy": stratum_accuracy,
        "balanced_stratum_accuracy": float(np.mean(list(stratum_accuracy.values()))),
        "coarse_question_family_macro_accuracy": float(
            np.mean([np.mean(scores) for scores in families.values()])
        ),
        "mean_input_tokens": float(np.mean([row["input_tokens"] for row in values])),
        "mean_output_tokens": float(np.mean([row["output_tokens"] for row in values])),
        "mean_evidence_units": float(np.mean([row["evidence_unit_count"] for row in values])),
        "mean_evidence_characters": float(np.mean([row["evidence_character_count"] for row in values])),
        "provenance_complete_rate": float(np.mean([row["provenance_complete"] for row in values])),
    }


def _case_bootstrap(
    left: dict[tuple[str, int], dict[str, Any]],
    right: dict[tuple[str, int], dict[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if set(left) != set(right):
        raise RuntimeError("Compared V17 conditions do not contain identical questions")
    by_case: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in sorted(left):
        by_case[key[0]].append(key)
    case_ids = sorted(by_case)
    per_case = []
    for case_id in case_ids:
        keys = by_case[case_id]
        left_exact = sum(set(left[key]["gold_indices"]) == set(left[key]["predicted_indices"]) for key in keys)
        right_exact = sum(set(right[key]["gold_indices"]) == set(right[key]["predicted_indices"]) for key in keys)
        per_case.append((len(keys), left_exact, right_exact))
    array = np.asarray(per_case, dtype=np.float64)
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = array[rng.integers(0, len(array), size=len(array))]
        differences[index] = (sampled[:, 1].sum() - sampled[:, 2].sum()) / sampled[:, 0].sum()
    observed = (_metrics(left.values())["exact_answer_set_accuracy"] - _metrics(right.values())["exact_answer_set_accuracy"])
    return {
        "case_count": len(case_ids),
        "replicates": replicates,
        "seed": seed,
        "exact_accuracy_difference": float(observed),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "bootstrap_probability_difference_gt_zero": float(np.mean(differences > 0)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    config = _load_json(args.config)
    manifest = _load_json(args.pilot_manifest)
    if manifest.get("test_accessed") is not False or manifest.get("data_role") != "final_qa_calibration":
        raise RuntimeError("V17 generation manifest violates the sealed-Test boundary")
    retrieval_rows = {
        (str(row["case_id"]), int(row["question_index"])): row
        for row in _read_jsonl(args.retrieval_rows)
    }
    selected = _selected_questions(args.radrestruct_root, set(manifest["case_ids"]))
    if len(selected) != int(manifest["question_count"]):
        raise RuntimeError("Pilot manifest question count does not match Rad-ReStruct")
    if args.limit_questions is not None:
        selected = selected[: int(args.limit_questions)]
    if any((row["case_id"], row["question_index"]) not in retrieval_rows for row in selected):
        raise RuntimeError("Retrieval rows do not cover the frozen generation pilot")

    raw_cases = {str(row["case_id"]): row for row in _read_jsonl(args.cases)}
    radgraph = read_radgraph_case_records(args.radgraph)
    facts_by_case = {case_id: tuple(record.facts) for case_id, record in radgraph.items() if record.status == "ok"}
    existing = {str(row["run_key"]): row for row in _read_jsonl(args.rows_output)}
    if args.reuse_no_history_rows is not None and not any(
        key.startswith("no_history|") for key in existing
    ):
        reusable = [
            row
            for row in _read_jsonl(args.reuse_no_history_rows)
            if row.get("condition") == "no_history"
            and str(row.get("case_id")) in set(manifest["case_ids"])
        ]
        _append_rows(args.rows_output, reusable)
        existing.update({str(row["run_key"]): row for row in reusable})
    evidence_cache: dict[tuple[str, str, int], tuple[list[str], list[dict[str, Any]], dict[str, float]]] = {}

    for row in selected:
        key = (row["case_id"], row["question_index"])
        retrieval = retrieval_rows[key]
        if "control_rankings" not in retrieval:
            raise RuntimeError(f"V17 retrieval row lacks matched controls: {key}")
        plan = plan_question(row["question"], raw_cases[row["case_id"]].get("indication", ""))
        evidence_cache[("no_history", *key)] = ([], [], evidence_profile([]))
        for condition in ("related", "random", "mismatched"):
            case_ids = [str(value) for value in retrieval["control_rankings"][condition]]
            if config["evidence"]["selector"] == "whole_report_top1":
                evidence_cache[(condition, *key)] = _whole_report(raw_cases[case_ids[0]])
                continue
            hierarchical = select_hierarchical_evidence(
                [raw_cases[case_id] for case_id in case_ids],
                query=row["question"],
                facts_by_case={case_id: facts_by_case.get(case_id, ()) for case_id in case_ids},
                plan=plan,
                maximum_cases=int(config["evidence"]["maximum_cases"]),
                maximum_units_per_case=int(config["evidence"]["maximum_units_per_case"]),
                maximum_total_units=int(config["evidence"]["maximum_total_units"]),
                maximum_characters=int(config["evidence"]["maximum_characters"]),
            )
            records = hierarchical.as_records()
            evidence_cache[(condition, *key)] = (
                [f"{unit.provenance_id} | {unit.text}" for unit in hierarchical.units],
                records,
                evidence_profile(hierarchical.units),
            )

    generator = MedGemmaImageGenerator(cache_dir=args.cache_dir, local_files_only=bool(config["model"]["local_files_only"]))
    adapter_value = config["model"].get("adapter")
    adapter_dir = args.adapter_dir or (ROOT / str(adapter_value) if adapter_value else None)
    model_arm = "base"
    if adapter_dir is not None:
        from peft import PeftModel

        if not (adapter_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(adapter_dir / "adapter_config.json")
        generator.model = PeftModel.from_pretrained(
            generator.model, str(adapter_dir), is_trainable=False
        )
        generator.model.eval()
        model_arm = "qlora_384"
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for condition in config["conditions"]:
        pending = [
            row for row in selected
            if f"{condition}|{row['case_id']}|{row['question_index']}" not in existing
        ]
        for offset in range(0, len(pending), int(config["model"]["batch_size"])):
            batch = pending[offset : offset + int(config["model"]["batch_size"])]
            prompts: list[str] = []
            image_paths: list[Path] = []
            evidence_rows = []
            for row in batch:
                evidence = evidence_cache[(condition, row["case_id"], row["question_index"])]
                evidence_rows.append(evidence)
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
                max_new_tokens=int(config["model"]["max_new_tokens"]),
                stop_token=str(config["model"]["stop_token"]),
            )
            completed = []
            for row, output, evidence in zip(batch, generated, evidence_rows, strict=True):
                parsed = parse_option_indices_with_wrapper_repair(
                    output["answer"], option_count=len(row["options"]), answer_type=row["answer_type"]
                )
                gold = [index for index, option in enumerate(row["options"]) if option in set(row["gold_answers"])]
                retrieval = retrieval_rows[(row["case_id"], row["question_index"])]
                run_key = f"{condition}|{row['case_id']}|{row['question_index']}"
                record = {
                    "run_key": run_key,
                    "condition": condition,
                    "case_id": row["case_id"],
                    "question_index": row["question_index"],
                    "answer_type": row["answer_type"],
                    "path": row["path"],
                    "stratum": retrieval["stratum"],
                    "gold_indices": gold,
                    "predicted_indices": parsed["indices"],
                    "contract_valid": parsed["contract_valid"],
                    "repairs": parsed["repairs"],
                    "raw_output": parsed["raw_output"],
                    "input_tokens": int(output["input_tokens"]),
                    "output_tokens": int(output["output_tokens"]),
                    "hit_token_ceiling": bool(output["hit_token_ceiling"]),
                    "evidence_unit_count": int(evidence[2]["unit_count"]),
                    "evidence_character_count": int(evidence[2]["character_count"]),
                    "evidence_case_ids": sorted({item["case_id"] for item in evidence[1]}),
                    "provenance_complete": bool(evidence[2]["provenance_complete_rate"] == 1.0),
                }
                completed.append(record)
                existing[run_key] = record
            _append_rows(args.rows_output, completed)

    expected = {
        f"{condition}|{row['case_id']}|{row['question_index']}"
        for condition in config["conditions"] for row in selected
    }
    if not expected <= set(existing):
        raise RuntimeError("V17 generation output is incomplete")
    all_rows = [existing[key] for key in sorted(expected)]
    by_condition = {
        condition: {
            (row["case_id"], int(row["question_index"])): row
            for row in all_rows if row["condition"] == condition
        }
        for condition in config["conditions"]
    }
    metrics = {condition: _metrics(rows.values()) for condition, rows in by_condition.items()}
    comparisons = {
        f"related_minus_{condition}": _case_bootstrap(
            by_condition["related"],
            by_condition[condition],
            seed=int(config["seed"]),
            replicates=int(config["evaluation"]["bootstrap_replicates"]),
        )
        for condition in ("no_history", "random", "mismatched")
    }
    no_history = by_condition["no_history"]
    negative_transfer = {}
    for condition in ("related", "random", "mismatched"):
        eligible = sum(set(row["gold_indices"]) == set(row["predicted_indices"]) for row in no_history.values())
        count = sum(
            set(no_history[key]["gold_indices"]) == set(no_history[key]["predicted_indices"])
            and set(by_condition[condition][key]["gold_indices"]) != set(by_condition[condition][key]["predicted_indices"])
            for key in no_history
        )
        negative_transfer[condition] = {"count": count, "baseline_correct_denominator": eligible, "rate": count / eligible if eligible else None}
    related_strata = metrics["related"]["stratum_exact_accuracy"]
    control_best_positive = max(metrics[name]["stratum_exact_accuracy"].get("positive", 0.0) for name in ("random", "mismatched"))
    control_best_nonbinary = max(metrics[name]["stratum_exact_accuracy"].get("non_binary", 0.0) for name in ("random", "mismatched"))
    success_checks = {
        "primary_exact_above_random": metrics["related"]["exact_answer_set_accuracy"] > metrics["random"]["exact_answer_set_accuracy"],
        "primary_exact_above_mismatched": metrics["related"]["exact_answer_set_accuracy"] > metrics["mismatched"]["exact_answer_set_accuracy"],
        "related_balanced_above_random": metrics["related"]["balanced_stratum_accuracy"] > metrics["random"]["balanced_stratum_accuracy"],
        "related_balanced_above_mismatched": metrics["related"]["balanced_stratum_accuracy"] > metrics["mismatched"]["balanced_stratum_accuracy"],
        "gain_not_negative_only": related_strata.get("positive", 0.0) > control_best_positive or related_strata.get("non_binary", 0.0) > control_best_nonbinary,
    }
    summary = {
        "study": "V17 matched historical-evidence MedGemma Calibration pilot",
        "status": "generation_pilot_success" if all(success_checks.values()) else "generation_pilot_mixed_or_negative",
        "data_role": "final_qa_calibration",
        "test_accessed": False,
        "model_arm": model_arm,
        "adapter_dir": str(adapter_dir.resolve()) if adapter_dir is not None else None,
        "case_count": len({row["case_id"] for row in selected}),
        "question_count_per_condition": len(selected),
        "condition_metrics": metrics,
        "comparisons": comparisons,
        "negative_transfer_from_no_history": negative_transfer,
        "success_checks": success_checks,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "peak_vram_mb_this_invocation": torch.cuda.max_memory_allocated() / 1024**2,
        "boundary": config["boundary"],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/v17_generation_pilot.json")
    parser.add_argument("--pilot-manifest", type=Path, default=ROOT / "data/splits/final_qa/v17_generation_pilot_manifest.json")
    parser.add_argument("--retrieval-rows", type=Path, default=ROOT / "experiments/v17_exploratory/v17_retrieval_calibration_rows.jsonl")
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--radgraph", type=Path, default=ROOT / "data/processed/v9_radgraph_modern_xl.jsonl")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/openi_official_images")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".hf_cache")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--reuse-no-history-rows", type=Path)
    parser.add_argument("--rows-output", type=Path, default=ROOT / "experiments/v17_exploratory/v17_generation_pilot_rows.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "experiments/v17_exploratory/v17_generation_pilot_summary.json")
    parser.add_argument("--limit-questions", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
