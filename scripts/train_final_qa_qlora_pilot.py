"""Train the frozen Train-only Final-QA MedGemma QLoRA pilot.

The foundation model remains frozen. Only q/v LoRA parameters are optimized,
and the internal monitoring split is case-disjoint from the optimization rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from train_v16_qlora import (
    MODEL_NAME,
    MODEL_REVISION,
    file_sha256,
    forward_loss,
    is_sequence_limit_error,
    load_peft_model,
    prepare_example,
    seed_everything,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINING_SEED = 7026
INTERNAL_SPLIT_SEED = 7031
INTERNAL_FRACTION = 0.10
CONDITIONS = ("no_history", "random_history", "relevant_fact_history")
EXPECTED_STRATA = {
    "binary_yes": 160,
    "binary_no": 160,
    "single_choice_nonbinary": 70,
    "multi_choice": 180,
    "fixed_choice": 30,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def internal_fraction(case_id: str) -> float:
    payload = f"final-qa-qlora-internal|{INTERNAL_SPLIT_SEED}|{case_id}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def row_key(row: Mapping[str, Any]) -> str:
    return f"{row['case_id']}|{row['question_index']}|{row['condition']}"


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 1800:
        raise RuntimeError(f"Expected 1800 pilot rows, found {len(rows)}")
    keys = [row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Final-QA QLoRA rows contain duplicate condition keys")
    condition_counts = {
        condition: sum(row["condition"] == condition for row in rows)
        for condition in CONDITIONS
    }
    if condition_counts != {condition: 600 for condition in CONDITIONS}:
        raise RuntimeError(f"Unexpected condition counts: {condition_counts}")
    no_history = [row for row in rows if row["condition"] == "no_history"]
    strata = {
        stratum: sum(row["stratum"] == stratum for row in no_history)
        for stratum in EXPECTED_STRATA
    }
    if strata != EXPECTED_STRATA:
        raise RuntimeError(f"Unexpected no-history strata: {strata}")


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Final-QA QLoRA requires CUDA")
    seed_everything(args.seed)
    rows = read_jsonl(args.examples)
    validate_rows(rows)
    case_ids = sorted({str(row["case_id"]) for row in rows})
    internal_cases = {
        case_id for case_id in case_ids if internal_fraction(case_id) < INTERNAL_FRACTION
    }
    train_rows = [row for row in rows if str(row["case_id"]) not in internal_cases]
    internal_rows = [row for row in rows if str(row["case_id"]) in internal_cases]
    if not train_rows or not internal_rows:
        raise RuntimeError("Case-level internal split produced an empty partition")

    model, processor, model_stats = load_peft_model(args.cache_dir, ("q_proj", "v_proj"))
    model.train()
    torch.cuda.reset_peak_memory_stats()
    frozen_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)

    losses: list[float] = []
    internal_losses: list[float] = []
    skipped_train: list[str] = []
    skipped_internal: list[str] = []
    start = time.perf_counter()
    steps = 0
    epoch = 0
    while steps < args.max_steps:
        order = list(range(len(train_rows)))
        random.Random(args.seed + epoch).shuffle(order)
        for row_index in order:
            if steps >= args.max_steps:
                break
            row = train_rows[row_index]
            try:
                batch = prepare_example(processor, row)
            except RuntimeError as error:
                if not is_sequence_limit_error(error):
                    raise
                skipped_train.append(row_key(row))
                continue
            loss = forward_loss(model, batch)
            (loss / args.gradient_accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            steps += 1
            if steps % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if steps % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "forward_step": steps,
                            "tail_loss": float(np.mean(losses[-args.log_every :])),
                            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
                        }
                    ),
                    flush=True,
                )
        epoch += 1

    if steps % args.gradient_accumulation:
        raise RuntimeError("Frozen max_steps must be divisible by gradient accumulation")
    model.eval()
    with torch.no_grad():
        for row in internal_rows:
            if len(internal_losses) >= args.internal_eval_examples:
                break
            try:
                batch = prepare_example(processor, row)
            except RuntimeError as error:
                if not is_sequence_limit_error(error):
                    raise
                skipped_internal.append(row_key(row))
                continue
            internal_losses.append(float(forward_loss(model, batch).detach().cpu()))
    if not losses or not internal_losses:
        raise RuntimeError("Training or internal monitoring produced no usable losses")
    if any(not parameter.requires_grad and parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("A frozen foundation-model parameter received a gradient")

    adapter_dir = args.output_dir / "adapter"
    processor_dir = args.output_dir / "processor"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(processor_dir)
    summary = {
        "study": "Final QA balanced-history MedGemma QLoRA pilot",
        "status": "train_internal_monitoring_complete",
        "model": {"name": MODEL_NAME, "revision": MODEL_REVISION},
        "inputs": {
            "examples_sha256": file_sha256(args.examples),
            "row_count": len(rows),
            "case_count": len(case_ids),
            "train_row_count": len(train_rows),
            "internal_row_count": len(internal_rows),
            "train_case_count": len(set(case_ids) - internal_cases),
            "internal_case_count": len(internal_cases),
            "internal_case_ids_sha256": hashlib.sha256(
                "\n".join(sorted(internal_cases)).encode("utf-8")
            ).hexdigest(),
            "skipped_training_row_count": len(set(skipped_train)),
            "skipped_internal_row_count": len(set(skipped_internal)),
        },
        "configuration": {
            "seed": args.seed,
            "internal_split_seed": INTERNAL_SPLIT_SEED,
            "internal_fraction": INTERNAL_FRACTION,
            "conditions": list(CONDITIONS),
            "learning_rate": args.learning_rate,
            "gradient_accumulation": args.gradient_accumulation,
            "max_forward_steps": args.max_steps,
            "quantization": "4bit_nf4_double_quant_bfloat16",
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_suffixes": ["q_proj", "v_proj"],
        },
        "model_statistics": {**model_stats, "frozen_parameter_count": frozen_parameter_count},
        "training": {
            "epochs_entered": epoch,
            "forward_steps": steps,
            "optimizer_steps": steps // args.gradient_accumulation,
            "mean_tail_loss": float(np.mean(losses[-min(100, len(losses)) :])),
            "internal_loss_mean": float(np.mean(internal_losses)),
            "internal_eval_examples": len(internal_losses),
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "elapsed_seconds": time.perf_counter() - start,
        },
        "artifacts": {
            "adapter_dir": str(adapter_dir.resolve()),
            "adapter_config_sha256": file_sha256(adapter_dir / "adapter_config.json"),
        },
        "claim_boundary": (
            "Train-only adaptation with case-level internal monitoring; no Calibration, "
            "Validation, Test, clinical-accuracy, or positive-effect claim."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples",
        type=Path,
        default=ROOT / "experiments/final_qa_development/final_qa_qlora_pilot_examples.jsonl",
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".hf_cache")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/final_qa_development/final_qa_qlora_pilot",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "experiments/final_qa_development/final_qa_qlora_training_summary.json",
    )
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--internal-eval-examples", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=16)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
