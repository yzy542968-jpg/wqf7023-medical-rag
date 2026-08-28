"""Train a small, leakage-controlled V16 MedGemma QLoRA adapter.

This trainer intentionally keeps the loop explicit so the image/text
processor, supervised-token mask, train/internal split, and frozen-parameter
checks are auditable without depending on a high-level training framework.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402


MODEL_NAME = "google/medgemma-1.5-4b-it"
MODEL_REVISION = "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
TRAINING_SEED = 1616
INTERNAL_SPLIT_SEED = 1618
INTERNAL_FRACTION = 0.10
MAX_SEQUENCE_LENGTH = 768
TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
TARGET_PROFILES = {
    "qv": ("q_proj", "v_proj"),
    "all": TARGET_SUFFIXES,
}
CONDITIONS = ("no_history", "retrieved_history", "random_history")
QUESTION_TYPES = ("findings", "impression")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_fraction(case_id: str) -> float:
    digest = hashlib.sha256(
        f"v16-internal-early-stop|{INTERNAL_SPLIT_SEED}|{case_id}".encode("utf-8")
    ).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def stable_case_order(case_ids: Sequence[str], seed: int) -> list[str]:
    return sorted(
        (str(case_id) for case_id in case_ids),
        key=lambda case_id: (
            hashlib.sha256(f"v16-development-subset|{seed}|{case_id}".encode("utf-8")).hexdigest(),
            case_id,
        ),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def render_messages(processor: Any, prompt: str, answer: str) -> tuple[str, str]:
    user = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": str(prompt)},
            ],
        }
    ]
    full = [*user, {"role": "assistant", "content": str(answer)}]
    prompt_text = processor.apply_chat_template(
        user,
        add_generation_prompt=True,
        tokenize=False,
    )
    full_text = processor.apply_chat_template(
        full,
        add_generation_prompt=False,
        tokenize=False,
    )
    return prompt_text, full_text


def prepare_example(processor: Any, row: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    image_path = Path(str(row["image_path"]))
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image = Image.open(image_path).convert("RGB")
    try:
        prompt_text, full_text = render_messages(processor, str(row["prompt"]), str(row["answer"]))
        prompt_inputs = processor(
            text=prompt_text,
            images=[[image]],
            return_tensors="pt",
            padding=False,
        )
        full_inputs = processor(
            text=full_text,
            images=[[image]],
            return_tensors="pt",
            padding=False,
        )
    finally:
        image.close()
    input_ids = full_inputs["input_ids"]
    attention_mask = full_inputs["attention_mask"]
    prompt_length = int(prompt_inputs["attention_mask"].sum().item())
    sequence_length = int(input_ids.shape[-1])
    if prompt_length >= sequence_length:
        raise RuntimeError(
            f"No supervised answer tokens for {row.get('case_id')}/{row.get('question_type')}"
        )
    if sequence_length > MAX_SEQUENCE_LENGTH:
        raise RuntimeError(
            f"Sequence length {sequence_length} exceeds V16 limit {MAX_SEQUENCE_LENGTH} "
            f"for {row.get('case_id')}/{row.get('question_type')}"
        )
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    labels[attention_mask == 0] = -100
    result: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    for key, value in full_inputs.items():
        if key not in result and isinstance(value, torch.Tensor):
            result[key] = value
    return result


def move_to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def is_sequence_limit_error(error: RuntimeError) -> bool:
    return "exceeds V16 limit" in str(error)


def row_key(row: Mapping[str, Any]) -> str:
    return f"{row['case_id']}|{row['question_type']}|{row['condition']}"


def load_peft_model(
    cache_dir: Path, target_suffixes: Sequence[str]
) -> tuple[Any, Any, dict[str, Any]]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    local_model_path = (
        cache_dir
        / "models--google--medgemma-1.5-4b-it"
        / "snapshots"
        / MODEL_REVISION
    )
    model_source = str(local_model_path) if local_model_path.is_dir() else MODEL_NAME
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
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        # Keep compatibility with older Transformers versions used by clones.
        model.gradient_checkpointing_enable()
    target_modules = [
        name
        for name, module in model.named_modules()
        if name.startswith("model.language_model.layers.")
        and name.rsplit(".", 1)[-1] in set(target_suffixes)
        and isinstance(module, torch.nn.Module)
    ]
    if not target_modules:
        raise RuntimeError("Could not find language-model target modules for LoRA")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    # Point PEFT at the local snapshot so saving an adapter never attempts a
    # network lookup for the gated base-model config.
    model.peft_config["default"].base_model_name_or_path = model_source
    model.enable_input_require_grads()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return model, processor, {
        "target_module_count": len(target_modules),
        "trainable_parameter_count": trainable,
        "total_parameter_count": total,
        "trainable_fraction": trainable / total,
    }


def forward_loss(model: Any, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    device = model.device
    moved = move_to_device(batch, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**moved)
    loss = output.loss
    if not torch.isfinite(loss):
        raise RuntimeError("V16 encountered a non-finite training loss")
    return loss


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("V16 QLoRA requires CUDA")
    seed_everything(args.seed)
    rows = read_jsonl(args.examples)
    if not rows:
        raise RuntimeError("V16 examples are empty")
    case_ids = sorted({str(row["case_id"]) for row in rows})
    if args.max_cases is not None:
        selected = set(stable_case_order(case_ids, args.seed)[: args.max_cases])
        rows = [row for row in rows if str(row["case_id"]) in selected]
        case_ids = sorted(selected)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    rows = [row for row in rows if str(row.get("condition")) in set(args.conditions)]
    rows = [row for row in rows if str(row.get("question_type")) in set(args.question_types)]
    if not rows:
        raise RuntimeError("No V16 rows remain after limits")
    internal_cases = {case_id for case_id in case_ids if stable_fraction(case_id) < INTERNAL_FRACTION}
    train_rows = [row for row in rows if str(row["case_id"]) not in internal_cases]
    internal_rows = [row for row in rows if str(row["case_id"]) in internal_cases]
    if not train_rows or not internal_rows:
        raise RuntimeError("V16 internal split is empty; increase the case limit")

    model, processor, model_stats = load_peft_model(
        args.cache_dir, TARGET_PROFILES[args.target_profile]
    )
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    frozen_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    losses: list[float] = []
    internal_losses: list[float] = []
    skipped_training_rows: list[str] = []
    skipped_internal_rows: list[str] = []
    start_time = time.perf_counter()
    steps = 0
    for epoch in range(args.epochs):
        order = list(range(len(train_rows)))
        random.Random(args.seed + epoch).shuffle(order)
        for row_index in order:
            row = train_rows[row_index]
            try:
                batch = prepare_example(processor, row)
            except RuntimeError as error:
                if not is_sequence_limit_error(error):
                    raise
                skipped_training_rows.append(row_key(row))
                continue
            loss = forward_loss(model, batch)
            (loss / args.gradient_accumulation).backward()
            if (steps + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            steps += 1
            if steps % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": steps,
                            "loss": float(np.mean(losses[-args.log_every :])),
                            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
                        }
                    ),
                    flush=True,
                )
            if args.max_steps is not None and steps >= args.max_steps:
                break
        if args.max_steps is not None and steps >= args.max_steps:
            break

    model.eval()
    with torch.no_grad():
        evaluated = 0
        for row in internal_rows:
            if evaluated >= args.internal_eval_examples:
                break
            try:
                batch = prepare_example(processor, row)
            except RuntimeError as error:
                if not is_sequence_limit_error(error):
                    raise
                skipped_internal_rows.append(row_key(row))
                continue
            internal_losses.append(float(forward_loss(model, batch).detach().cpu()))
            evaluated += 1
    if not losses:
        raise RuntimeError("V16 has no trainable examples after sequence filtering")
    if any(
        parameter.requires_grad is False and parameter.grad is not None
        for parameter in model.parameters()
    ):
        raise RuntimeError("A frozen V16 parameter unexpectedly received a gradient")
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(args.output_dir / "processor")
    summary = {
        "study": "V16 MedGemma QLoRA development",
        "status": args.run_label,
        "model": {"name": MODEL_NAME, "revision": MODEL_REVISION},
        "inputs": {
            "examples_path": str(args.examples.resolve()),
            "examples_sha256": file_sha256(args.examples),
            "row_count": len(rows),
            "case_count": len(case_ids),
            "train_row_count": len(train_rows),
            "internal_row_count": len(internal_rows),
            "train_case_count": len(set(case_ids) - internal_cases),
            "internal_case_count": len(internal_cases),
            "frozen_parameter_count": frozen_parameter_count,
            "internal_cases_sha256": hashlib.sha256("\n".join(sorted(internal_cases)).encode("utf-8")).hexdigest(),
            "skipped_training_row_count": len(set(skipped_training_rows)),
            "skipped_training_rows_sha256": hashlib.sha256(
                "\n".join(sorted(set(skipped_training_rows))).encode("utf-8")
            ).hexdigest(),
            "skipped_internal_row_count": len(set(skipped_internal_rows)),
            "skipped_internal_rows_sha256": hashlib.sha256(
                "\n".join(sorted(set(skipped_internal_rows))).encode("utf-8")
            ).hexdigest(),
        },
        "configuration": {
            "seed": args.seed,
            "internal_split_seed": INTERNAL_SPLIT_SEED,
            "internal_fraction": INTERNAL_FRACTION,
            "conditions": list(args.conditions),
            "question_types": list(args.question_types),
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "gradient_accumulation": args.gradient_accumulation,
            "quantization": "4bit_nf4_double_quant_bfloat16",
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_profile": args.target_profile,
            "target_suffixes": list(TARGET_PROFILES[args.target_profile]),
        },
        "model_statistics": model_stats,
        "training": {
            "optimizer_steps": steps // args.gradient_accumulation,
            "forward_steps": steps,
            "mean_tail_loss": float(np.mean(losses[-min(len(losses), 100) :])),
            "internal_loss_mean": float(np.mean(internal_losses)),
            "internal_eval_examples": len(internal_losses),
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "elapsed_seconds": time.perf_counter() - start_time,
        },
        "artifacts": {
            "adapter_dir": str(adapter_dir.resolve()),
            "adapter_config_sha256": file_sha256(adapter_dir / "adapter_config.json"),
        },
        "claim_boundary": "Train/internal loss smoke or development evidence only; no clinical accuracy claim.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=ROOT / "experiments/v16_adaptation/v16_sft_examples.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".hf_cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/v16_adaptation/qlora_smoke")
    parser.add_argument("--summary", type=Path, default=ROOT / "data/splits/v16/v16_qlora_smoke_summary.json")
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--internal-eval-examples", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--question-types", nargs="+", choices=QUESTION_TYPES, default=list(QUESTION_TYPES))
    parser.add_argument("--target-profile", choices=tuple(TARGET_PROFILES), default="all")
    parser.add_argument("--run-label", default="development_pilot")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
