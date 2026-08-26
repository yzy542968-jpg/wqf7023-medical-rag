"""Verify the Qwen3 cache signature without loading any model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402
from medical_rag.similar_case.radgraph_adapter import read_radgraph_case_records  # noqa: E402
from medical_rag.similar_case.v10_runtime import QUESTIONS  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402

CACHE_SCHEMA = "v12-qwen3-last-token-512-v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--radgraph", type=Path, default=ROOT / "data/processed/v9_radgraph_modern_xl.jsonl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/v10/v10_cluster_disjoint_split.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_qwen3_v10_embeddings.npz")
    args = parser.parse_args()
    raw_cases = {
        str(row["case_id"]): row
        for row in (
            json.loads(line)
            for line in args.cases.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    formal = {
        case.study_id: case
        for case in read_openi_paired_cases(args.cases, source_unique_patient=True, radgraph_path=args.radgraph)
    }
    radgraph = read_radgraph_case_records(args.radgraph)
    split = read_json(args.split)
    train_ids = [str(value) for value in split["partitions"]["train"]["case_ids"]]
    validation_ids = [str(value) for value in split["partitions"]["validation"]["case_ids"]]
    eligible = {case_id for case_id in train_ids + validation_ids if case_id in formal and radgraph[case_id].status == "ok"}
    train_ids = [case_id for case_id in train_ids if case_id in eligible]
    validation_ids = [case_id for case_id in validation_ids if case_id in eligible]
    all_ids = train_ids + validation_ids
    query_keys = [f"{case_id}:{question_type}" for case_id in all_ids for question_type in QUESTIONS]
    query_texts = [
        "\n".join(part for part in (formal[case_id].indication, QUESTIONS[question_type]) if part)
        for case_id in all_ids
        for question_type in QUESTIONS
    ]
    document_texts = [formal[case_id].report_text for case_id in train_ids]
    signature_payload = {
        "cache_schema": CACHE_SCHEMA,
        "cases_sha256": file_sha256(args.cases),
        "candidate_ids": train_ids,
        "query_keys": query_keys,
        "query_text_sha256": hashlib.sha256("\n\u241e\n".join(query_texts).encode("utf-8")).hexdigest(),
        "document_text_sha256": hashlib.sha256("\n\u241e\n".join(document_texts).encode("utf-8")).hexdigest(),
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "instruction": "Given a radiology question and clinical indication, retrieve a chest X-ray report containing clinically relevant evidence for the question.",
        "max_seq_length": 512,
        "pooling": "last_non_padding_token",
    }
    expected = hashlib.sha256(json.dumps(signature_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    with np.load(args.cache, allow_pickle=False) as cache:
        actual = str(cache["signature"].item())
        shapes = {key: list(cache[key].shape) for key in ("document_embeddings", "query_embeddings")}
    print(json.dumps({"expected": expected, "actual": actual, "match": expected == actual, "train": len(train_ids), "validation": len(validation_ids), "shapes": shapes, "document_sha": signature_payload["document_text_sha256"], "query_sha": signature_payload["query_text_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
