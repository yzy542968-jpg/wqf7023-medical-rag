"""Cache pinned MedSigLIP embeddings for unique Final-QA Validation questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.multimodal.medsiglip import MedSiglipEncoder  # noqa: E402
from medical_rag.qa.radrestruct import iter_radrestruct_cases  # noqa: E402


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _text_fingerprint(texts: list[str]) -> str:
    payload = "\n".join(texts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_json(args.config)
    manifest = _load_json(args.manifest)
    validation_ids = {
        str(case["case_id"]) for case in manifest["roles"]["validation"]["cases"]
    }
    cases = {
        case.case_id: case
        for case in iter_radrestruct_cases(args.radrestruct_root)
        if case.case_id in validation_ids
    }
    missing = sorted(validation_ids - set(cases))
    if missing:
        raise ValueError(f"Missing {len(missing)} Validation cases in Rad-ReStruct")

    all_questions = [
        question.question
        for case_id in sorted(cases)
        for question in cases[case_id].questions
    ]
    unique_questions = sorted(set(all_questions))
    encoder_config = config["question_encoder"]
    started = time.perf_counter()
    encoder = MedSiglipEncoder(
        str(encoder_config["model"]),
        revision=str(encoder_config["revision"]),
        device=str(encoder_config["device"]),
        cache_dir=ROOT / ".hf_cache",
        max_text_tokens=int(encoder_config["max_tokens"]),
        local_files_only=bool(args.local_files_only),
    )
    embeddings = encoder.encode_texts(
        unique_questions,
        batch_size=int(encoder_config["batch_size"]),
    ).astype(np.float32, copy=False)
    if embeddings.shape[0] != len(unique_questions):
        raise RuntimeError("Question embedding row count is inconsistent")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Question embeddings contain non-finite values")

    fingerprint = _text_fingerprint(unique_questions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        questions=np.asarray(unique_questions),
        embeddings=embeddings,
        text_sha256=np.asarray(fingerprint),
        model=np.asarray(str(encoder_config["model"])),
        revision=np.asarray(str(encoder_config["revision"])),
    )
    summary = {
        "study": config["study"],
        "role": "Final-QA Validation reused as development only",
        "test_accessed": False,
        "device": str(encoder_config["device"]),
        "model": str(encoder_config["model"]),
        "revision": str(encoder_config["revision"]),
        "case_count": len(cases),
        "question_instance_count": len(all_questions),
        "unique_question_count": len(unique_questions),
        "embedding_dimension": int(embeddings.shape[1]),
        "question_text_sha256": fingerprint,
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(args.output.relative_to(ROOT)).replace("\\", "/"),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/final_qa_v2_selective_gate.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json",
    )
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments/final_qa_development/final_qa_v2_question_embeddings.npz",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT
        / "experiments/final_qa_development/final_qa_v2_question_embeddings.json",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
