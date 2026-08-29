from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.multimodal.v9_generation import select_primary_image  # noqa: E402
from medical_rag.qa.medgemma_contract import build_compact_qa_prompt  # noqa: E402
from medical_rag.qa.radrestruct import iter_radrestruct_cases  # noqa: E402
from medical_rag.similar_case.radgraph_adapter import read_radgraph_case_records  # noqa: E402
from medical_rag.similar_case.v11_evidence import select_hierarchical_evidence  # noqa: E402
from medical_rag.similar_case.v11_question_planner import plan_question  # noqa: E402


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _hash(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stratum(question: Any) -> str:
    if question.answer_type == "fixed_choice":
        return "fixed_choice"
    if question.answer_type == "multi_choice":
        return "multi_choice"
    if question.answer_type == "single_choice" and question.options == ("yes", "no"):
        if question.answers == ("yes",):
            return "binary_yes"
        if question.answers == ("no",):
            return "binary_no"
    if question.answer_type == "single_choice":
        return "single_choice_nonbinary"
    raise ValueError(f"Unsupported QA stratum: {question.answer_type}/{question.options}")


def _embedding_map(path: Path) -> tuple[dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as payload:
        ids = [str(value) for value in payload["case_ids"]]
        values = np.asarray(payload["case_image_embeddings"], dtype=np.float32)
        signature = str(payload["signature"].item())
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return dict(zip(ids, values, strict=True)), signature


def _whole_report(case: dict[str, Any]) -> list[str]:
    output = []
    for section in ("findings", "impression"):
        text = " ".join(str(case.get(section) or "").split())
        if text:
            output.append(f"{case['case_id']} | {section} | {text}")
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_json(args.config)
    manifest = _load_json(args.manifest)
    train_manifest = manifest["roles"]["train"]["cases"]
    train_ids = {case["case_id"] for case in train_manifest}
    cluster_by_id = {case["case_id"]: case["cluster_id"] for case in train_manifest}
    raw_cases = {str(row["case_id"]): row for row in _read_jsonl(args.cases)}
    embeddings, embedding_signature = _embedding_map(args.embeddings)
    radgraph = read_radgraph_case_records(args.radgraph)

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in iter_radrestruct_cases(args.radrestruct_root):
        if case.case_id not in train_ids or case.case_id not in embeddings:
            continue
        for question_index, question in enumerate(case.questions):
            candidates[_stratum(question)].append(
                {
                    "case_id": case.case_id,
                    "question_index": question_index,
                    "question": question,
                }
            )
    selected: list[dict[str, Any]] = []
    for stratum, quota in config["base_question_sampling"].items():
        if stratum in {"method", "total"}:
            continue
        if stratum not in candidates or not isinstance(quota, int):
            raise ValueError(f"Unsupported sampling quota entry: {stratum}={quota!r}")
        ranked = sorted(
            candidates[stratum],
            key=lambda row: (
                _hash("final-qa-qlora-base", config["seed"], row["case_id"], row["question_index"]),
                row["case_id"],
                row["question_index"],
            ),
        )
        if len(ranked) < int(quota):
            raise ValueError(f"Not enough {stratum}: {len(ranked)} < {quota}")
        for row in ranked[: int(quota)]:
            row["stratum"] = stratum
            selected.append(row)
    if len(selected) != int(config["base_question_sampling"]["total"]):
        raise RuntimeError("Base-question quota total is inconsistent")

    bank_ids = sorted(
        case_id
        for case_id in train_ids
        if case_id in raw_cases
        and case_id in embeddings
        and case_id in radgraph
        and radgraph[case_id].status == "ok"
        and (raw_cases[case_id].get("findings") or raw_cases[case_id].get("impression"))
    )
    bank_matrix = np.stack([embeddings[case_id] for case_id in bank_ids])
    top3_by_target: dict[str, list[str]] = {}
    random_by_target: dict[str, str] = {}
    for target_id in sorted({row["case_id"] for row in selected}):
        eligible = [
            index
            for index, case_id in enumerate(bank_ids)
            if case_id != target_id and cluster_by_id[case_id] != cluster_by_id[target_id]
        ]
        eligible_set = set(eligible)
        scores = embeddings[target_id] @ bank_matrix.T
        scores[[index for index in range(len(bank_ids)) if index not in eligible_set]] = -np.inf
        top3_by_target[target_id] = [bank_ids[index] for index in np.argsort(-scores, kind="stable")[:3]]
        random_by_target[target_id] = min(
            (bank_ids[index] for index in eligible),
            key=lambda case_id: _hash("final-qa-qlora-random", config["seed"], target_id, case_id),
        )

    examples: list[dict[str, Any]] = []
    leakage_count = 0
    same_cluster_history_count = 0
    for row in sorted(selected, key=lambda value: (value["case_id"], value["question_index"])):
        target_id = row["case_id"]
        question = row["question"]
        target = raw_cases[target_id]
        top_ids = top3_by_target[target_id]
        plan = plan_question(question.question, target.get("indication", ""))
        hierarchical = select_hierarchical_evidence(
            [raw_cases[case_id] for case_id in top_ids],
            query=question.question,
            facts_by_case={case_id: tuple(radgraph[case_id].facts) for case_id in top_ids},
            plan=plan,
            maximum_cases=3,
            maximum_units_per_case=2,
            maximum_total_units=6,
            maximum_characters=1200,
        )
        evidence_by_condition = {
            "no_history": ([], []),
            "random_history": (
                _whole_report(raw_cases[random_by_target[target_id]]),
                [random_by_target[target_id]],
            ),
            "relevant_fact_history": (
                [f"{unit.provenance_id} | {unit.text}" for unit in hierarchical.units],
                list(hierarchical.retrieved_case_ids),
            ),
        }
        gold_indices = [
            index
            for index, option in enumerate(question.options)
            if option in set(question.answers)
        ]
        answer = json.dumps(gold_indices, separators=(",", ":"))
        for condition in config["conditions_per_question"]:
            evidence, history_ids = evidence_by_condition[condition]
            if any(cluster_by_id[history_id] == cluster_by_id[target_id] for history_id in history_ids):
                same_cluster_history_count += 1
            prompt = build_compact_qa_prompt(
                question=question.question,
                options=question.options,
                indication=target.get("indication"),
                image_available=True,
                historical_evidence=evidence,
            )
            # The prompt builder receives indication, question, options and the
            # explicit historical evidence list only. Target report fields are
            # never passed to it; target-case presence is audited separately.
            leakage_count += int(target_id in history_ids)
            examples.append(
                {
                    "case_id": target_id,
                    "cluster_id": cluster_by_id[target_id],
                    "question_index": row["question_index"],
                    "question_type": question.answer_type,
                    "stratum": row["stratum"],
                    "condition": condition,
                    "image_path": str(select_primary_image(target, args.image_root)),
                    "prompt": prompt,
                    "answer": answer,
                    "history_case_ids": history_ids,
                }
            )
    if leakage_count or same_cluster_history_count:
        raise RuntimeError(
            f"SFT leakage audit failed: target_report={leakage_count}, same_cluster={same_cluster_history_count}"
        )
    if len(examples) != int(config["expected_example_count"]):
        raise RuntimeError("Unexpected SFT example count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in examples),
        encoding="utf-8",
    )
    summary = {
        "study": config["study"],
        "status": "train_only_sft_dataset_complete",
        "example_count": len(examples),
        "base_question_count": len(selected),
        "base_question_strata": dict(Counter(row["stratum"] for row in selected)),
        "examples_by_condition": dict(Counter(row["condition"] for row in examples)),
        "case_count": len({row["case_id"] for row in examples}),
        "historical_bank_case_count": len(bank_ids),
        "same_cluster_history_count": same_cluster_history_count,
        "target_report_prompt_leakage_count": leakage_count,
        "embedding_signature": embedding_signature,
        "output_sha256": _file_sha256(args.output),
        "output_local_only": str(args.output.relative_to(ROOT)),
        "boundary": config["boundary"],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/final_qa_qlora_pilot.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json")
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/processed/v10_medsiglip_embeddings.npz")
    parser.add_argument("--radgraph", type=Path, default=ROOT / "data/processed/v9_radgraph_modern_xl.jsonl")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/openi_official_images")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/final_qa_development/final_qa_qlora_pilot_examples.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "experiments/final_qa_development/final_qa_qlora_pilot_dataset_summary.json")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
