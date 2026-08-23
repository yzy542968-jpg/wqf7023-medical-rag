from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.multimodal.evaluation import aggregate_case_images  # noqa: E402
from medical_rag.multimodal.fusion import l2_normalize  # noqa: E402
from medical_rag.multimodal.medsiglip import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    MedSiglipEncoder,
)
from medical_rag.multimodal.openi_images import resolve_official_image  # noqa: E402
from medical_rag.multimodal.v6_chunking import build_report_chunks  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402


DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v10" / "v10_cluster_disjoint_split.json"
DEFAULT_IMAGES = ROOT / "data" / "raw" / "openi_official_images"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "v10_medsiglip_embeddings.npz"
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v10" / "v10_medsiglip_embedding_summary.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def image_lookup(image_root: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in image_root.rglob("*.png"):
        if path.name in lookup:
            raise RuntimeError(f"Duplicate official image name: {path.name}")
        lookup[path.name] = path
    return lookup


def resolve_views(
    case_ids: Sequence[str],
    cases: Mapping[str, Mapping[str, Any]],
    lookup: Mapping[str, Path],
) -> tuple[list[str], list[Path]]:
    view_case_ids: list[str] = []
    view_paths: list[Path] = []
    for case_id in case_ids:
        paths = [
            resolve_official_image(case_id, str(image["filename"]), lookup)
            for image in cases[case_id].get("images", [])
        ]
        resolved = [path for path in paths if path is not None]
        if not resolved:
            raise RuntimeError(f"V10 case {case_id} has no readable image")
        view_case_ids.extend([case_id] * len(resolved))
        view_paths.extend(resolved)
    return view_case_ids, view_paths


def aggregate_report_chunks(
    chunk_embeddings: np.ndarray,
    chunk_case_ids: Sequence[str],
    report_ids: Sequence[str],
) -> np.ndarray:
    grouped: dict[str, list[np.ndarray]] = {case_id: [] for case_id in report_ids}
    for embedding, case_id in zip(chunk_embeddings, chunk_case_ids, strict=True):
        grouped[case_id].append(np.asarray(embedding, dtype=np.float32))
    if any(not grouped[case_id] for case_id in report_ids):
        raise RuntimeError("Every report-bearing case must have at least one chunk")
    return np.stack(
        [l2_normalize(np.stack(grouped[case_id]).mean(axis=0)) for case_id in report_ids]
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode the complete V10 OpenI source with MedSigLIP.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--text-batch-size", type=int, default=32)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    rows = read_jsonl(args.cases)
    cases = {str(row["case_id"]): row for row in rows}
    split = json.loads(args.split.read_text(encoding="utf-8"))
    case_ids = sorted(cases)
    partition_ids = {
        case_id
        for partition in split["partitions"].values()
        for case_id in partition["case_ids"]
    }
    if set(case_ids) != partition_ids:
        raise RuntimeError("V10 split and source case universe differ")

    lookup = image_lookup(args.image_root)
    view_case_ids, view_paths = resolve_views(case_ids, cases, lookup)
    report_ids = sorted(
        case_id
        for case_id, row in cases.items()
        if str(row.get("findings", "")).strip() or str(row.get("impression", "")).strip()
    )

    encoder = MedSiglipEncoder(
        revision=DEFAULT_REVISION,
        cache_dir=ROOT / ".hf_cache",
        local_files_only=True,
    )
    chunks = [
        chunk
        for case_id in report_ids
        for chunk in build_report_chunks(cases[case_id], encoder.processor.tokenizer, max_tokens=64)
    ]
    chunk_case_ids = [str(chunk["case_id"]) for chunk in chunks]
    chunk_texts = [str(chunk["text"]) for chunk in chunks]
    signature = stable_signature(
        {
            "runner_sha256": file_sha256(Path(__file__)),
            "encoder_sha256": file_sha256(ROOT / "src" / "medical_rag" / "multimodal" / "medsiglip.py"),
            "chunker_sha256": file_sha256(ROOT / "src" / "medical_rag" / "multimodal" / "v6_chunking.py"),
            "cases_sha256": file_sha256(args.cases),
            "split_sha256": file_sha256(args.split),
            "model": DEFAULT_MODEL,
            "revision": DEFAULT_REVISION,
            "case_ids": case_ids,
            "report_ids": report_ids,
            "chunk_texts_sha256": hashlib.sha256("\n".join(chunk_texts).encode("utf-8")).hexdigest(),
            "view_paths": [str(path.relative_to(args.image_root)) for path in view_paths],
        }
    )
    if args.output.exists():
        with np.load(args.output, allow_pickle=False) as cached:
            if str(cached["signature"].item()) == signature:
                print(json.dumps({"status": "cache_hit", "signature": signature}, indent=2))
                return

    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    view_embeddings = encoder.encode_images(view_paths, batch_size=args.image_batch_size)
    case_image_embeddings = aggregate_case_images(view_embeddings, view_case_ids, case_ids)
    chunk_embeddings = encoder.encode_texts(chunk_texts, batch_size=args.text_batch_size)
    report_embeddings = aggregate_report_chunks(chunk_embeddings, chunk_case_ids, report_ids)
    report_index = {case_id: index for index, case_id in enumerate(report_ids)}
    chunk_case_indices = np.asarray([report_index[case_id] for case_id in chunk_case_ids], dtype=np.int32)
    elapsed = time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated() / (1024**2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        signature=np.asarray(signature),
        case_ids=np.asarray(case_ids),
        view_case_ids=np.asarray(view_case_ids),
        view_paths=np.asarray([str(path.relative_to(args.image_root)) for path in view_paths]),
        view_embeddings=view_embeddings.astype(np.float32),
        case_image_embeddings=case_image_embeddings.astype(np.float32),
        report_ids=np.asarray(report_ids),
        report_embeddings=report_embeddings.astype(np.float32),
        chunk_case_indices=chunk_case_indices,
        chunk_embeddings=chunk_embeddings.astype(np.float32),
        chunk_text_sha256=np.asarray(
            [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in chunk_texts]
        ),
        elapsed_seconds=np.asarray(elapsed),
        peak_gpu_memory_mib=np.asarray(peak_mib),
    )
    summary = {
        "study": "V10 complete-source MedSigLIP encoding",
        "status": "complete",
        "model": DEFAULT_MODEL,
        "revision": DEFAULT_REVISION,
        "signature": signature,
        "artifact_sha256": file_sha256(args.output),
        "case_count": len(case_ids),
        "view_count": len(view_paths),
        "report_count": len(report_ids),
        "chunk_count": len(chunks),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_mib": peak_mib,
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

