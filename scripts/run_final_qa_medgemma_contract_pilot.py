from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.multimodal.v9_generation import (  # noqa: E402
    MedGemmaImageGenerator,
    select_primary_image,
)
from medical_rag.qa.medgemma_contract import (  # noqa: E402
    build_compact_qa_prompt,
    parse_option_indices,
)
from medical_rag.qa.radrestruct import iter_radrestruct_cases  # noqa: E402


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _selection_hash(domain: str, seed: int, case_id: str, index: int) -> str:
    payload = f"{domain}|{seed}|{case_id}|{index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _select_rows(
    config: dict[str, Any],
    manifest: dict[str, Any],
    radrestruct_root: Path,
) -> list[dict[str, Any]]:
    calibration_ids = {
        case["case_id"] for case in manifest["roles"]["calibration"]["cases"]
    }
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in iter_radrestruct_cases(radrestruct_root):
        if case.case_id not in calibration_ids:
            continue
        for index, question in enumerate(case.questions):
            candidates[question.answer_type].append(
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
    sampling = config["sampling"]
    selected: list[dict[str, Any]] = []
    for answer_type in ("single_choice", "multi_choice", "fixed_choice"):
        quota = int(sampling[answer_type])
        ranked = sorted(
            candidates[answer_type],
            key=lambda row: _selection_hash(
                sampling["domain"],
                int(config["seed"]),
                row["case_id"],
                int(row["question_index"]),
            ),
        )
        if len(ranked) < quota:
            raise ValueError(f"Not enough {answer_type} rows for pilot quota {quota}")
        selected.extend(ranked[:quota])
    return sorted(
        selected, key=lambda row: (row["case_id"], row["question_index"])
    )


def _existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = _read_jsonl(path)
    return {str(row["run_key"]): row for row in rows}


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    true_positive = false_positive = false_negative = 0
    by_type: dict[str, list[float]] = defaultdict(list)
    exact_values: list[float] = []
    valid_values: list[float] = []
    for row in rows:
        gold = set(row["gold_indices"])
        predicted = set(row["predicted_indices"])
        exact = float(gold == predicted)
        exact_values.append(exact)
        valid_values.append(float(row["contract_valid"]))
        by_type[row["answer_type"]].append(exact)
        true_positive += len(gold & predicted)
        false_positive += len(predicted - gold)
        false_negative += len(gold - predicted)
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "row_count": len(rows),
        "exact_answer_set_accuracy": sum(exact_values) / len(exact_values),
        "option_micro_f1": 2 * true_positive / denominator if denominator else 0.0,
        "contract_valid_rate": sum(valid_values) / len(valid_values),
        "single_choice_accuracy": sum(by_type["single_choice"]) / len(by_type["single_choice"]),
        "multi_choice_exact_accuracy": sum(by_type["multi_choice"]) / len(by_type["multi_choice"]),
        "fixed_choice_accuracy": sum(by_type["fixed_choice"]) / len(by_type["fixed_choice"]),
        "mean_input_tokens": sum(row["input_tokens"] for row in rows) / len(rows),
        "mean_output_tokens": sum(row["output_tokens"] for row in rows) / len(rows),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    config = _load_json(args.config)
    manifest = _load_json(args.manifest)
    selected = _select_rows(config, manifest, args.radrestruct_root)
    raw_cases = {str(row["case_id"]): row for row in _read_jsonl(args.cases)}
    existing = _existing_rows(args.rows_output)
    generator = MedGemmaImageGenerator(local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    for condition in config["conditions"]:
        image_available = condition.startswith("b3_")
        pending = [
            row
            for row in selected
            if f"{condition}|{row['case_id']}|{row['question_index']}" not in existing
        ]
        for offset in range(0, len(pending), int(config["model"]["batch_size"])):
            batch = pending[offset : offset + int(config["model"]["batch_size"])]
            prompts: list[str] = []
            image_paths: list[Path] = []
            for row in batch:
                case = raw_cases[row["case_id"]]
                prompts.append(
                    build_compact_qa_prompt(
                        question=row["question"],
                        options=row["options"],
                        indication=case.get("indication") if image_available else None,
                        image_available=image_available,
                    )
                )
                if image_available:
                    image_paths.append(select_primary_image(case, args.image_root))
            if image_available:
                generated = generator.generate_batch(
                    prompts,
                    image_paths,
                    max_new_tokens=int(config["model"]["max_new_tokens"]),
                )
            else:
                generated = generator.generate_text_batch(
                    prompts,
                    max_new_tokens=int(config["model"]["max_new_tokens"]),
                )
            completed: list[dict[str, Any]] = []
            for row, output in zip(batch, generated, strict=True):
                parsed = parse_option_indices(
                    output["answer"],
                    option_count=len(row["options"]),
                    answer_type=row["answer_type"],
                )
                gold_indices = [
                    index
                    for index, option in enumerate(row["options"])
                    if option in set(row["gold_answers"])
                ]
                run_key = f"{condition}|{row['case_id']}|{row['question_index']}"
                record = {
                    "run_key": run_key,
                    "condition": condition,
                    "case_id": row["case_id"],
                    "question_index": row["question_index"],
                    "answer_type": row["answer_type"],
                    "path": row["path"],
                    "option_count": len(row["options"]),
                    "gold_indices": gold_indices,
                    "predicted_indices": parsed["indices"],
                    "contract_valid": parsed["contract_valid"],
                    "raw_output": parsed["raw_output"],
                    "input_tokens": int(output["input_tokens"]),
                    "output_tokens": int(output["output_tokens"]),
                }
                completed.append(record)
                existing[run_key] = record
            _append_rows(args.rows_output, completed)

    elapsed = time.perf_counter() - start
    selected_keys = {
        f"{condition}|{row['case_id']}|{row['question_index']}"
        for condition in config["conditions"]
        for row in selected
    }
    complete_rows = [existing[key] for key in sorted(selected_keys)]
    expected = len(selected) * len(config["conditions"])
    if len(complete_rows) != expected:
        raise RuntimeError(f"Pilot is incomplete: {len(complete_rows)} of {expected}")
    by_condition = {
        condition: _summarize(
            [row for row in complete_rows if row["condition"] == condition]
        )
        for condition in config["conditions"]
    }
    summary = {
        "study": config["study"],
        "status": "calibration_contract_pilot_complete_no_test",
        "config": "config/final_qa_medgemma_contract_pilot_r1.json",
        "selected_row_count": len(selected),
        "conditions": by_condition,
        "elapsed_seconds_this_invocation": elapsed,
        "peak_vram_mb_this_invocation": torch.cuda.max_memory_allocated() / 1024**2,
        "rows_output_local_only": str(args.rows_output.relative_to(ROOT)),
        "boundary": config["boundary"],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/final_qa_medgemma_contract_pilot_r1.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json",
    )
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl"
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=ROOT / "data/raw/openi_official_images",
    )
    parser.add_argument(
        "--rows-output",
        type=Path,
        default=ROOT
        / "experiments/final_qa_development/medgemma_contract_pilot_rows.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=ROOT
        / "experiments/final_qa_development/medgemma_contract_pilot_summary.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
