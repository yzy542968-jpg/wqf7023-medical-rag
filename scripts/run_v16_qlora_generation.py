"""Run paired V16 generation with a frozen MedGemma base or QLoRA adapter.

The script reuses a preselected Validation manifest and saved V12 rankings.
It never reads the V10 Test partition and emits rows that can be compared
with the same prompts, images, and historical-case conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_v16_sft_dataset import (  # noqa: E402
    CONDITIONS,
    QUESTIONS,
    SELECTION_SEED,
    build_prompt,
    canonical_text,
    stable_rank,
)
from medical_rag.evaluation.answer_metrics import token_f1  # noqa: E402
from medical_rag.multimodal.v9_generation import select_primary_image  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from run_v12_generation_pilot import (  # noqa: E402
    bound_complete_sentences,
    read_json,
    read_jsonl,
    sha256_ids,
)


MODEL_NAME = "google/medgemma-1.5-4b-it"
MODEL_REVISION = "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
LOCAL_MODEL_DIR = (
    ROOT
    / ".hf_cache/models--google--medgemma-1.5-4b-it/snapshots/"
    / MODEL_REVISION
)
DEFAULT_EXPECTED_CASE_COUNT = 48
HISTORY_TOP_K = 3


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def digest_ids(values: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted({str(value).strip() for value in values})).encode("utf-8")
    ).hexdigest()


def load_model(cache_dir: Path, adapter_dir: Path | None) -> tuple[Any, Any, str]:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    model_source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.is_dir() else MODEL_NAME
    kwargs = {
        "revision": MODEL_REVISION,
        "cache_dir": str(cache_dir),
        "local_files_only": True,
    }
    processor = AutoProcessor.from_pretrained(model_source, use_fast=False, **kwargs)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_source,
        quantization_config=quantization,
        device_map="auto",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        **kwargs,
    )
    model.config.use_cache = True
    arm = "base"
    if adapter_dir is not None:
        if not (adapter_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(f"QLoRA adapter config is missing: {adapter_dir}")
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        arm = "qlora"
    model.eval()
    return model, processor, arm


def render_prompt(processor: Any, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def generate_one(
    model: Any,
    processor: Any,
    prompt: str,
    image_path: Path,
    *,
    max_new_tokens: int,
    answer_sentence_limit: int = 2,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    try:
        rendered = render_prompt(processor, prompt)
        inputs = processor(
            text=[rendered],
            images=[[image]],
            padding=True,
            return_tensors="pt",
        )
    finally:
        image.close()
    device = model.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_tokens = int(inputs["attention_mask"].sum().item())
    stop_token_id = int(processor.tokenizer.convert_tokens_to_ids("<end_of_turn>"))
    if stop_token_id == processor.tokenizer.unk_token_id:
        stop_token_id = None
    eos_token_id = model.generation_config.eos_token_id
    if stop_token_id is not None:
        eos_values = [] if eos_token_id is None else (
            list(eos_token_id) if isinstance(eos_token_id, (list, tuple)) else [int(eos_token_id)]
        )
        eos_token_id = sorted(set([*eos_values, stop_token_id]))
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
        )
    prompt_width = int(inputs["input_ids"].shape[1])
    answer_ids = generated[0, prompt_width:]
    raw = processor.decode(answer_ids, skip_special_tokens=True).strip()
    for marker in ("<end_of_turn>", "<eos>", "</s>"):
        raw = raw.replace(marker, " ")
    raw = " ".join(raw.split())
    answer = bound_complete_sentences(raw, maximum=answer_sentence_limit).strip()
    return {
        "answer": answer,
        "raw_output": raw,
        "input_tokens": input_tokens,
        "output_tokens": int(answer_ids.shape[-1]),
        "hit_token_ceiling": float(answer_ids.shape[-1] >= max_new_tokens),
        "answer_only_contract_valid": float(bool(answer) and not any(
            marker in raw.lower() for marker in ('"answer"', "analysis:", "json")
        )),
    }


def generate_batch(
    model: Any,
    processor: Any,
    tasks: Sequence[Mapping[str, Any]],
    *,
    max_new_tokens: int,
    answer_sentence_limit: int = 2,
) -> list[dict[str, Any]]:
    images = [Image.open(Path(str(task["target_image_path"]))).convert("RGB") for task in tasks]
    try:
        rendered = [render_prompt(processor, str(task["prompt"])) for task in tasks]
        inputs = processor(
            text=rendered,
            images=[[image] for image in images],
            padding=True,
            return_tensors="pt",
        )
    finally:
        for image in images:
            image.close()
    device = model.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_tokens = inputs["attention_mask"].sum(dim=1).tolist()
    stop_token_id = int(processor.tokenizer.convert_tokens_to_ids("<end_of_turn>"))
    if stop_token_id == processor.tokenizer.unk_token_id:
        stop_token_id = None
    eos_token_id = model.generation_config.eos_token_id
    if stop_token_id is not None:
        eos_values = [] if eos_token_id is None else (
            list(eos_token_id) if isinstance(eos_token_id, (list, tuple)) else [int(eos_token_id)]
        )
        eos_token_id = sorted(set([*eos_values, stop_token_id]))
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
        )
    prompt_width = int(inputs["input_ids"].shape[1])
    results: list[dict[str, Any]] = []
    for row, input_length in zip(generated, input_tokens, strict=True):
        answer_ids = row[prompt_width:]
        raw = processor.decode(answer_ids, skip_special_tokens=True).strip()
        for marker in ("<end_of_turn>", "<eos>", "</s>"):
            raw = raw.replace(marker, " ")
        raw = " ".join(raw.split())
        answer = bound_complete_sentences(raw, maximum=answer_sentence_limit).strip()
        results.append(
            {
                "answer": answer,
                "raw_output": raw,
                "input_tokens": int(input_length),
                "output_tokens": int(answer_ids.shape[-1]),
                "hit_token_ceiling": float(answer_ids.shape[-1] >= max_new_tokens),
                "answer_only_contract_valid": float(bool(answer) and not any(
                    marker in raw.lower() for marker in ('"answer"', "analysis:", "json")
                )),
            }
        )
    return results


def cluster_map(split: Mapping[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for cluster in split["clusters"]:
        for case_id in cluster["case_ids"]:
            output[str(case_id)] = str(cluster["cluster_id"])
    return output


def build_tasks(
    cases: Mapping[str, Mapping[str, Any]],
    selected_case_ids: Sequence[str],
    ranking_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    clusters: Mapping[str, str],
    image_root: Path,
    *,
    conditions: Sequence[str],
    max_new_tokens: int,
    history_top_k: int,
    answer_sentence_limit: int,
) -> list[dict[str, Any]]:
    if history_top_k < 1:
        raise ValueError("history_top_k must be positive")
    if answer_sentence_limit < 1:
        raise ValueError("answer_sentence_limit must be positive")
    all_ranked_ids: set[str] = set()
    for row in ranking_rows.values():
        for key in ("full_bank_lambdamart", "rrf_lambdamart"):
            all_ranked_ids.update(str(value) for value in row["rankings"].get(key, []))
    tasks: list[dict[str, Any]] = []
    for case_id in selected_case_ids:
        source = cases[case_id]
        excluded = {
            candidate_id
            for candidate_id in all_ranked_ids
            if clusters.get(candidate_id) == clusters.get(case_id)
        }
        eligible_random = [
            candidate_id
            for candidate_id in all_ranked_ids
            if candidate_id not in excluded and candidate_id != case_id and candidate_id in cases
        ]
        for question_type, question in QUESTIONS.items():
            if question_type not in {"findings", "impression"}:
                continue
            ranking = ranking_rows[(case_id, question_type)]["rankings"]
            retrieved_ids = [
                str(value)
                for value in ranking["rrf_lambdamart"]
                if str(value) not in excluded and str(value) != case_id
            ][:history_top_k]
            random_ids = stable_rank(
                eligible_random,
                "v16-validation-random-history",
                str(SELECTION_SEED),
                case_id,
                question_type,
            )[:history_top_k]
            histories = {
                "no_history": [],
                "retrieved_history": [cases[value] for value in retrieved_ids],
                "random_history": [cases[value] for value in random_ids],
            }
            image_path = select_primary_image(source, image_root)
            for condition in conditions:
                history = histories[condition]
                prompt = build_prompt(
                    indication=source.get("indication", ""),
                    question=question,
                    question_type=question_type,
                    retrieved_cases=history,
                )
                prompt = prompt.replace(
                    "at most two concise complete sentences",
                    f"at most {answer_sentence_limit} concise complete sentences",
                )
                reference = canonical_text(source.get(question_type, ""))
                tasks.append(
                    {
                        "case_id": case_id,
                        "question_type": question_type,
                        "question": question,
                        "condition": condition,
                        "reference_answer": reference,
                        "retrieved_case_ids": [str(value["case_id"]) for value in history],
                        "target_image_path": str(image_path),
                        "prompt": prompt,
                        "max_new_tokens": max_new_tokens,
                        "history_top_k": history_top_k,
                        "answer_sentence_limit": answer_sentence_limit,
                    }
                )
    return tasks


def run(args: argparse.Namespace) -> None:
    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    split = read_json(args.split)
    clusters = cluster_map(split)
    selection_rows = read_jsonl(args.selection_rows)
    selected_case_ids = sorted({str(row["case_id"]) for row in selection_rows})
    expected_case_count = int(args.expected_case_count)
    if expected_case_count < 1:
        raise ValueError("expected_case_count must be positive")
    if len(selected_case_ids) != expected_case_count:
        raise RuntimeError(
            "V16 selection manifest has "
            f"{len(selected_case_ids)} cases; expected {expected_case_count}"
        )
    selected_case_set = set(selected_case_ids)
    ranking_rows = {
        (str(row["case_id"]), str(row["question_type"])): row
        for row in read_jsonl(args.ranking_rows)
        if str(row["case_id"]) in selected_case_set
        and str(row["question_type"]) in {"findings", "impression"}
    }
    expected_ranking_keys = {
        (case_id, question_type)
        for case_id in selected_case_ids
        for question_type in ("findings", "impression")
    }
    if set(ranking_rows) != expected_ranking_keys:
        raise RuntimeError(
            "V16 ranking rows do not cover the requested selection manifest x 2 matrix"
        )
    tasks = build_tasks(
        cases,
        selected_case_ids,
        ranking_rows,
        clusters,
        args.image_root,
        conditions=tuple(args.conditions),
        max_new_tokens=args.max_new_tokens,
        history_top_k=args.history_top_k,
        answer_sentence_limit=args.answer_sentence_limit,
    )
    if args.max_cases is not None:
        keep = set(selected_case_ids[: args.max_cases])
        tasks = [task for task in tasks if task["case_id"] in keep]
    expected = min(args.max_cases or expected_case_count, expected_case_count) * 2 * len(args.conditions)
    if len(tasks) != expected:
        raise RuntimeError(f"V16 generation matrix is incomplete: {len(tasks)} != {expected}")
    for task in tasks:
        if not Path(task["target_image_path"]).is_file():
            raise FileNotFoundError(task["target_image_path"])

    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(args.rows_output) if args.rows_output.is_file() else []
    completed = {
        (str(row["case_id"]), str(row["question_type"]), str(row["condition"]))
        for row in existing
    }
    pending = [task for task in tasks if (
        str(task["case_id"]), str(task["question_type"]), str(task["condition"])
    ) not in completed]
    model, processor, arm = load_model(args.cache_dir, args.adapter_dir)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with args.rows_output.open("a", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(pending), max(1, args.batch_size)):
            batch = pending[start : start + max(1, args.batch_size)]
            batch_started = time.perf_counter()
            generated_batch = generate_batch(
                model,
                processor,
                batch,
                max_new_tokens=args.max_new_tokens,
                answer_sentence_limit=args.answer_sentence_limit,
            )
            latency = (time.perf_counter() - batch_started) / len(batch)
            for task, generated in zip(batch, generated_batch, strict=True):
                retrieved = set(task["retrieved_case_ids"])
                row = {
                    **task,
                    **generated,
                    "model_arm": arm,
                    "model_name": MODEL_NAME,
                    "model_revision": MODEL_REVISION,
                    "v16_prompt_version": "v16_sft_prompt_v1",
                    "evidence_provenance_valid": float(
                        (task["condition"] == "no_history" and not task["retrieved_case_ids"])
                        or (
                            task["condition"] != "no_history"
                            and len(task["retrieved_case_ids"]) == args.history_top_k
                            and len(retrieved) == args.history_top_k
                            and all(value in cases for value in task["retrieved_case_ids"])
                        )
                    ),
                    "token_f1": token_f1(generated["answer"], task["reference_answer"]),
                    "latency_seconds": latency,
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_gpu_memory_mib": (
                        torch.cuda.max_memory_allocated() / 1024**2
                        if torch.cuda.is_available()
                        else 0.0
                    ),
                }
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            done = min(start + len(batch), len(pending))
            if done % 8 == 0 or done == len(pending):
                print(json.dumps({"arm": arm, "generated": done, "pending": len(pending)}), flush=True)
    rows = read_jsonl(args.rows_output)
    expected_keys = {
        (str(task["case_id"]), str(task["question_type"]), str(task["condition"]))
        for task in tasks
    }
    observed_keys = {
        (str(row["case_id"]), str(row["question_type"]), str(row["condition"]))
        for row in rows
    }
    if observed_keys != expected_keys or len(rows) != len(expected_keys):
        raise RuntimeError("V16 generation rows are incomplete or duplicated")
    summary = {
        "study": "V16 QLoRA paired generation pilot",
        "status": "validation_generation_complete_no_retuning",
        "model_arm": arm,
        "no_test_evaluation": True,
        "counts": {
            "cases": len({str(row["case_id"]) for row in rows}),
            "rows": len(rows),
            "conditions": list(args.conditions),
            "selection_manifest_cases": expected_case_count,
        },
        "inputs": {
            "cases_sha256": file_sha256(args.cases),
            "split_sha256": file_sha256(args.split),
            "selection_rows_sha256": file_sha256(args.selection_rows),
            "selected_case_ids_sha256": sha256_ids(selected_case_ids),
            "ranking_rows_sha256": file_sha256(args.ranking_rows),
            "model_revision": MODEL_REVISION,
            "adapter_config_sha256": (
                file_sha256(args.adapter_dir / "adapter_config.json")
                if args.adapter_dir is not None
                else None
            ),
        },
        "configuration": {
            "conditions": list(args.conditions),
            "question_types": ["findings", "impression"],
            "history_top_k": args.history_top_k,
            "max_new_tokens": args.max_new_tokens,
            "answer_sentence_limit": args.answer_sentence_limit,
            "selection_seed": SELECTION_SEED,
            "expected_case_count": expected_case_count,
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mib": (
                torch.cuda.max_memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0.0
            ),
        },
        "rows_sha256": file_sha256(args.rows_output),
        "claim_boundary": (
            "Validation-only automated answer-reference consistency; not diagnostic "
            "accuracy, clinical safety, physician utility, or external validation."
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/v10/v10_cluster_disjoint_split.json")
    parser.add_argument("--ranking-rows", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_qwen3_validation_rankings_rows.jsonl")
    parser.add_argument("--selection-rows", type=Path, default=ROOT / "experiments/v12_optimization/generation/v12_generation_selection_rows.jsonl")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/openi_official_images")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".hf_cache")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--history-top-k", type=int, default=HISTORY_TOP_K)
    parser.add_argument("--answer-sentence-limit", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--expected-case-count",
        type=int,
        default=DEFAULT_EXPECTED_CASE_COUNT,
        help="Number of case IDs required in the supplied selection manifest.",
    )
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
