"""Build the V12 Qwen3 cache in a clean, model-only process.

The main retrieval pilot deliberately reads this cache instead of loading the
1.2GB Qwen3 model alongside RadGraph, LightGBM, and the multimodal runtime.
This keeps Windows/CUDA memory behavior predictable while preserving the exact
same candidate IDs, query strings, instruction, pooling, and normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
INSTRUCTION = (
    "Given a radiology question and clinical indication, retrieve a chest X-ray "
    "report containing clinically relevant evidence for the question."
)
MAX_SEQ_LENGTH = 512
CACHE_SCHEMA = "v12-qwen3-last-token-512-v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(cache_root: Path) -> Path:
    path = cache_root / "models--Qwen--Qwen3-Embedding-0.6B" / "snapshots" / MODEL_REVISION
    required = ("config.json", "model.safetensors", "tokenizer.json", "modules.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete Qwen3 snapshot {path}; missing={missing}")
    return path


def report_text(row: dict) -> str:
    return "\n".join(
        part
        for part in (
            " ".join(str(row.get("findings", "")).split()),
            " ".join(str(row.get("impression", "")).split()),
        )
        if part
    )


def normalized(value: object) -> str:
    return " ".join(str(value or "").split())


def encode_texts(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    values: Sequence[str],
    *,
    device: str,
    batch_size: int,
    label: str,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = list(values[start : start + batch_size])
        tokens = tokenizer(
            batch,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            result = model(**tokens)
            mask = tokens["attention_mask"]
            last_indices = mask.sum(dim=1) - 1
            pooled = result.last_hidden_state[
                torch.arange(result.last_hidden_state.shape[0], device=device),
                last_indices,
            ]
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        outputs.append(pooled.cpu().numpy().astype(np.float32, copy=False))
        del result, tokens, pooled
        if start == 0 or start + batch_size >= len(values) or (start + batch_size) % (batch_size * 20) == 0:
            print(f"{label}={min(start + batch_size, len(values))}/{len(values)}", flush=True)
    return np.concatenate(outputs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--radgraph", type=Path, default=ROOT / "data/processed/v9_radgraph_modern_xl.jsonl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/v10/v10_cluster_disjoint_split.json")
    parser.add_argument("--cache-root", type=Path, default=ROOT / ".hf_cache")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_qwen3_v10_embeddings.npz")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()
    raw_cases = {
        str(row["case_id"]): row
        for row in (
            json.loads(line)
            for line in args.cases.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    radgraph = read_jsonl_records(args.radgraph)
    split = read_json(args.split)
    train_ids = [str(case_id) for case_id in split["partitions"]["train"]["case_ids"]]
    validation_ids = [str(case_id) for case_id in split["partitions"]["validation"]["case_ids"]]
    train_ids = [case_id for case_id in train_ids if case_id in raw_cases and radgraph.get(case_id) == "ok"]
    validation_ids = [case_id for case_id in validation_ids if case_id in raw_cases and radgraph.get(case_id) == "ok"]
    all_ids = train_ids + validation_ids
    question_texts = {
        "findings": "What are the main radiographic findings?",
        "impression": "What is the most likely radiographic impression?",
        "acute": "Is there an acute cardiopulmonary abnormality? Explain briefly.",
    }
    query_keys = [f"{case_id}:{question_type}" for case_id in all_ids for question_type in question_texts]
    query_texts = [
        "\n".join(
            part
            for part in (normalized(raw_cases[case_id].get("indication", "")), question_texts[question_type])
            if part
        )
        for case_id in all_ids
        for question_type in question_texts
    ]
    document_texts = [report_text(raw_cases[case_id]) for case_id in train_ids]
    signature_payload = {
        "cache_schema": CACHE_SCHEMA,
        "cases_sha256": file_sha256(args.cases),
        "candidate_ids": train_ids,
        "query_keys": query_keys,
        "query_text_sha256": hashlib.sha256("\n\u241e\n".join(query_texts).encode("utf-8")).hexdigest(),
        "document_text_sha256": hashlib.sha256("\n\u241e\n".join(document_texts).encode("utf-8")).hexdigest(),
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "instruction": INSTRUCTION,
        "max_seq_length": MAX_SEQ_LENGTH,
        "pooling": "last_non_padding_token",
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    selected_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model_path = snapshot(args.cache_root)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, padding_side="left")
    model = AutoModel.from_pretrained(
        str(model_path),
        local_files_only=True,
        dtype=torch.float16 if selected_device == "cuda" else torch.float32,
    ).to(selected_device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    print(f"qwen3_model_loaded device={selected_device} train={len(train_ids)} validation={len(validation_ids)}", flush=True)
    instructed = [f"Instruct: {INSTRUCTION}\nQuery:{text}" for text in query_texts]
    documents = encode_texts(model, tokenizer, document_texts, device=selected_device, batch_size=max(1, args.batch_size), label="documents")
    queries = encode_texts(model, tokenizer, instructed, device=selected_device, batch_size=max(1, args.batch_size), label="queries")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    np.savez_compressed(
        temporary,
        signature=np.asarray(signature),
        document_embeddings=documents,
        query_embeddings=queries,
        build_seconds=np.asarray(time.perf_counter() - started),
        peak_gpu_memory_mib=np.asarray(float(torch.cuda.max_memory_allocated() / (1024**2)) if selected_device == "cuda" else 0.0),
    )
    os.replace(str(temporary) + ".npz", args.output)
    metadata = {
        "output": str(args.output.resolve().relative_to(ROOT)),
        "signature": signature,
        "sha256": file_sha256(args.output),
        "train_case_count": len(train_ids),
        "validation_case_count": len(validation_ids),
        "document_embedding_shape": list(documents.shape),
        "query_embedding_shape": list(queries.shape),
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "instruction": INSTRUCTION,
        "max_seq_length": MAX_SEQ_LENGTH,
        "pooling": "last_non_padding_token",
        "normalized": True,
        "device": selected_device,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(metadata, indent=2), flush=True)


def read_jsonl_records(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            records[str(row["case_id"])] = str(row["status"])
    return records


if __name__ == "__main__":
    main()
