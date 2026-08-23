from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from medical_rag.similar_case.openi_adapter import read_openi_paired_cases
from medical_rag.similar_case.radgraph_adapter import (
    RadGraphCaseRecord,
    read_radgraph_case_records,
)
from medical_rag.similar_case.schema import PairedCase
from medical_rag.similar_case.v10_multiview import (
    ViewAttention,
    attention_query_embedding,
    l2_normalize,
)
from medical_rag.similar_case.v10_runtime import FrozenR5Runtime


R5_SEEDS = (7041, 7042, 7043, 7044, 7045)
ATTENTION_SEEDS = (7051, 7052, 7053, 7054, 7055)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass
class V10RuntimeAssets:
    runtime: FrozenR5Runtime
    raw_cases: dict[str, dict[str, Any]]
    cases: dict[str, PairedCase]
    radgraph: dict[str, RadGraphCaseRecord]
    image_by_id: dict[str, np.ndarray]
    views_by_id: dict[str, np.ndarray]
    attention_models: list[ViewAttention]
    split: dict[str, Any]
    embedding_signature: str

    def attention_image(self, case_id: str) -> np.ndarray:
        return attention_query_embedding(self.attention_models, self.views_by_id[case_id])

    def partition_ids(self, partition: str) -> list[str]:
        identifiers = self.split["partitions"][partition]["case_ids"]
        return sorted(
            case_id
            for case_id in identifiers
            if case_id in self.cases
            and case_id in self.image_by_id
            and case_id in self.views_by_id
            and case_id in self.radgraph
            and self.radgraph[case_id].status == "ok"
        )


def load_v10_runtime_assets(
    *,
    cases_path: Path,
    radgraph_path: Path,
    split_path: Path,
    embeddings_path: Path,
    r5_checkpoint_dir: Path,
    attention_checkpoint_dir: Path,
) -> V10RuntimeAssets:
    raw_rows = read_jsonl(cases_path)
    raw_cases = {str(row["case_id"]): row for row in raw_rows}
    formal = read_openi_paired_cases(
        cases_path,
        source_unique_patient=True,
        radgraph_path=radgraph_path,
    )
    cases = {case.study_id: case for case in formal}
    radgraph = read_radgraph_case_records(radgraph_path)
    split = read_json(split_path)
    with np.load(embeddings_path, allow_pickle=False) as encoded:
        case_ids = [str(value) for value in encoded["case_ids"]]
        case_images = l2_normalize(np.asarray(encoded["case_image_embeddings"], dtype=np.float32))
        report_ids = [str(value) for value in encoded["report_ids"]]
        reports = l2_normalize(np.asarray(encoded["report_embeddings"], dtype=np.float32))
        view_ids = [str(value) for value in encoded["view_case_ids"]]
        views = l2_normalize(np.asarray(encoded["view_embeddings"], dtype=np.float32))
        embedding_signature = str(encoded["signature"].item())
    image_by_id = {case_id: case_images[index] for index, case_id in enumerate(case_ids)}
    report_by_id = {case_id: reports[index] for index, case_id in enumerate(report_ids)}
    view_lists: dict[str, list[np.ndarray]] = {}
    for case_id, embedding in zip(view_ids, views, strict=True):
        view_lists.setdefault(case_id, []).append(embedding)
    views_by_id = {case_id: np.stack(values) for case_id, values in view_lists.items()}

    eligible = {
        case_id
        for case_id in cases
        if case_id in image_by_id
        and case_id in report_by_id
        and case_id in views_by_id
        and radgraph[case_id].status == "ok"
    }
    candidate_ids = sorted(set(split["partitions"]["train"]["case_ids"]) & eligible)
    r5_states = [
        torch.load(
            r5_checkpoint_dir / f"r5_seed_{seed}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for seed in R5_SEEDS
    ]
    attention_models = []
    for seed in ATTENTION_SEEDS:
        model = ViewAttention()
        model.load_state_dict(
            torch.load(
                attention_checkpoint_dir / f"attention_seed_{seed}.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()
        attention_models.append(model)
    runtime = FrozenR5Runtime.build(
        candidate_ids=candidate_ids,
        cases=cases,
        raw_cases=raw_cases,
        facts_by_case={case_id: tuple(radgraph[case_id].facts) for case_id in candidate_ids},
        image_by_id=image_by_id,
        report_by_id=report_by_id,
        checkpoint_states=r5_states,
    )
    return V10RuntimeAssets(
        runtime=runtime,
        raw_cases=raw_cases,
        cases=cases,
        radgraph=radgraph,
        image_by_id=image_by_id,
        views_by_id=views_by_id,
        attention_models=attention_models,
        split=split,
        embedding_signature=embedding_signature,
    )


__all__ = [
    "ATTENTION_SEEDS",
    "R5_SEEDS",
    "V10RuntimeAssets",
    "load_v10_runtime_assets",
    "read_json",
    "read_jsonl",
]
