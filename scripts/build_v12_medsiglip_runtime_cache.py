"""Extract the V12 runtime MedSigLIP arrays into a small local cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed/v10_medsiglip_embeddings.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_medsiglip_runtime_embeddings.npz")
    args = parser.parse_args()
    with np.load(args.source, allow_pickle=False) as source:
        case_ids = np.asarray(source["case_ids"])
        case_images = np.asarray(source["case_image_embeddings"], dtype=np.float32)
        report_ids = np.asarray(source["report_ids"])
        reports = np.asarray(source["report_embeddings"], dtype=np.float32)
    if case_images.ndim != 2 or reports.ndim != 2 or case_images.shape[1] != reports.shape[1]:
        raise ValueError("MedSigLIP runtime embeddings have incompatible shapes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    np.savez_compressed(
        temporary,
        source_sha256=np.asarray(sha256(args.source)),
        case_ids=case_ids,
        case_image_embeddings=case_images,
        report_ids=report_ids,
        report_embeddings=reports,
    )
    os.replace(str(temporary) + ".npz", args.output)
    print(json.dumps({
        "output": str(args.output.resolve().relative_to(ROOT)),
        "source_sha256": sha256(args.source),
        "output_sha256": sha256(args.output),
        "case_image_shape": list(case_images.shape),
        "report_shape": list(reports.shape),
    }, indent=2))


if __name__ == "__main__":
    main()
