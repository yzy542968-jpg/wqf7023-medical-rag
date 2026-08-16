from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
from typing import Iterable


def _require_generation_stack():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional environment
        raise RuntimeError(
            "Hugging Face generation requires optional dependencies: torch and transformers."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _load_prompt_records(path: Path, max_items: int | None) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_items is not None and len(records) >= max_items:
                break
    return records


def _load_completed_qids(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            completed.add(json.loads(line)["qid"])
    return completed


def _deduplicate_existing_output(path: Path) -> int:
    if not path.exists():
        return 0
    unique: dict[str, dict] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        unique.setdefault(str(row["qid"]), row)
    temporary = path.with_suffix(path.suffix + ".deduplicated")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in unique.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return len(unique)


def _acquire_output_lock(path: Path) -> Path:
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Generation output is already locked: {lock_path}. "
            "Do not run concurrent writers for one JSONL file."
        ) from exc
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(str(os.getpid()))
    atexit.register(lock_path.unlink, missing_ok=True)
    return lock_path


def _format_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a careful medical question-answering assistant for a research experiment.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def _batched(records: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hugging Face LLM generation over a prompt pack.")
    parser.add_argument("--prompt-pack", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--max-new-tokens", default=256, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output file and skip prompt records whose qid is already present.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_output_lock(args.output)
    deduplicated_count = (
        _deduplicate_existing_output(args.output) if args.resume else 0
    )

    torch, AutoModelForCausalLM, AutoTokenizer = _require_generation_stack()
    selected_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if selected_device == "cuda" else torch.float32,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    ).to(selected_device)
    model.eval()

    records = _load_prompt_records(args.prompt_pack, args.max_items)
    completed_qids = _load_completed_qids(args.output) if args.resume else set()
    records_to_run = [record for record in records if record["qid"] not in completed_qids]

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as file:
        for batch in _batched(records_to_run, args.batch_size):
            prompt_texts = [
                _format_chat_prompt(tokenizer, record["prompt"]) for record in batch
            ]
            inputs = tokenizer(
                prompt_texts, return_tensors="pt", padding=True, truncation=True
            ).to(selected_device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_kwargs)
            prompt_length = inputs["input_ids"].shape[-1]
            for record, generated in zip(batch, output_ids, strict=True):
                answer = tokenizer.decode(
                    generated[prompt_length:], skip_special_tokens=True
                ).strip()
                output_record = {
                    "qid": record["qid"],
                    "case_id": record["case_id"],
                    "question_type": record.get("question_type"),
                    "system": record["system"],
                    "prompt_mode": record["prompt_mode"],
                    "retriever": record["retriever"],
                    "model": args.model,
                    "question": record["question"],
                    "answer": answer,
                    "reference_answer": record["reference_answer"],
                    "relevant_case_ids": record["relevant_case_ids"],
                    "retrieved_case_ids": record["retrieved_case_ids"],
                }
                file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            file.flush()

    print(
        json.dumps(
            {
                "output": str(args.output),
                "records_loaded": len(records),
                "records_skipped": len(completed_qids),
                "records_generated": len(records_to_run),
                "model": args.model,
                "batch_size": args.batch_size,
                "existing_unique_records": deduplicated_count,
            },
            indent=2,
        )
    )
    lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
