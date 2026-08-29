"""Build a V12 MedCPT query cache for a frozen split partition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from medical_rag.retrieval.medcpt_retriever import encode_queries  # noqa: E402
from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402
from medical_rag.similar_case.radgraph_adapter import read_radgraph_case_records  # noqa: E402
from medical_rag.similar_case.v10_runtime import QUESTIONS  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402

CACHE_SCHEMA = "v12-medcpt-query-cls-64-v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--radgraph", type=Path, default=ROOT / "data/processed/v9_radgraph_modern_xl.jsonl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/v10/v10_cluster_disjoint_split.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_medcpt_query_embeddings.npz")
    parser.add_argument(
        "--query-partition",
        choices=("calibration", "validation", "test"),
        default="validation",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    started = time.perf_counter()
    formal = {
        case.study_id: case
        for case in read_openi_paired_cases(args.cases, source_unique_patient=True, radgraph_path=args.radgraph)
    }
    radgraph = read_radgraph_case_records(args.radgraph)
    split = read_json(args.split)
    train_ids = [str(value) for value in split["partitions"]["train"]["case_ids"]]
    query_ids = [str(value) for value in split["partitions"][args.query_partition]["case_ids"]]
    eligible = {case_id for case_id in train_ids + query_ids if case_id in formal and radgraph[case_id].status == "ok"}
    train_ids = [case_id for case_id in train_ids if case_id in eligible]
    query_ids = [case_id for case_id in query_ids if case_id in eligible]
    all_ids = train_ids + query_ids
    query_keys = [f"{case_id}:{question_type}" for case_id in all_ids for question_type in QUESTIONS]
    query_texts = [
        "\n".join(part for part in (formal[case_id].indication, QUESTIONS[question_type]) if part)
        for case_id in all_ids
        for question_type in QUESTIONS
    ]
    payload = {
        "cache_schema": CACHE_SCHEMA,
        "cases_sha256": file_sha256(args.cases),
        "split_sha256": file_sha256(args.split),
        "candidate_ids": train_ids,
        "query_keys": query_keys,
        "query_text_sha256": hashlib.sha256("\n\u241e\n".join(query_texts).encode("utf-8")).hexdigest(),
        "model": "ncbi/MedCPT-Query-Encoder",
        "max_length": 64,
        "normalized": True,
    }
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"medcpt_query_start device={device} queries={len(query_texts)}", flush=True)
    embeddings = encode_queries(
        query_texts,
        batch_size=max(1, args.batch_size),
        device=device,
        max_length=64,
        local_files_only=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    np.savez_compressed(
        temporary,
        signature=np.asarray(signature),
        query_embeddings=np.asarray(embeddings, dtype=np.float32),
        query_keys=np.asarray(query_keys),
        build_seconds=np.asarray(time.perf_counter() - started),
    )
    os.replace(str(temporary) + ".npz", args.output)
    print(json.dumps({
        "output": str(args.output.resolve().relative_to(ROOT)),
        "signature": signature,
        "sha256": file_sha256(args.output),
        "shape": list(embeddings.shape),
        "train_case_count": len(train_ids),
        "query_partition": args.query_partition,
        "query_case_count": len(query_ids),
        "device": device,
        "elapsed_seconds": time.perf_counter() - started,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
