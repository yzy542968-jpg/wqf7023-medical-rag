from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hugging Face LLM generation over a prompt pack.")
    parser.add_argument("--prompt-pack", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--max-new-tokens", default=256, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output file and skip prompt records whose qid is already present.",
    )
    args = parser.parse_args()

    torch, AutoModelForCausalLM, AutoTokenizer = _require_generation_stack()
    selected_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
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
    args.output.parent.mkdir(parents=True, exist_ok=True)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as file:
        for record in records_to_run:
            prompt_text = _format_chat_prompt(tokenizer, record["prompt"])
            inputs = tokenizer(prompt_text, return_tensors="pt").to(selected_device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_kwargs)
            generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
