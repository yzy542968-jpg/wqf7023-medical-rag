from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.case_scoped_benchmark import (
    build_case_chunks,
    build_case_questions,
    is_clean_eligible_case,
)
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def case_ids_from_payload(payload: dict) -> set[str]:
    if "questions" in payload:
        return {str(row["case_id"]) for row in payload["questions"]}
    if "case_ids" in payload:
        return {str(value) for value in payload["case_ids"]}
    if "cases" in payload:
        return {str(row["case_id"]) for row in payload["cases"]}
    raise ValueError("Cannot find case IDs in prior manifest.")


def fingerprint(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fresh multimodal V5 OpenI cohort.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "multimodal_v5.json")
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "processed" / "openi_cases.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "openi_multimodal_v5_cohort.json")
    args = parser.parse_args()

    config = read_json(args.config)
    all_cases = load_cases_jsonl(args.cases)
    excluded: set[str] = set()
    for relative in config["cohort"]["excluded_source_manifests"]:
        path = ROOT / relative
        excluded.update(case_ids_from_payload(read_json(path)))

    eligible = [
        case for case in all_cases
        if str(case["case_id"]) not in excluded and is_clean_eligible_case(case)
    ]
    rng = random.Random(int(config["cohort"]["seed"]))
    rng.shuffle(eligible)
    selected = sorted(
        eligible[: int(config["cohort"]["selected_case_count"])],
        key=lambda row: str(row["case_id"]),
    )
    if len(selected) != int(config["cohort"]["selected_case_count"]):
        raise RuntimeError("Not enough fresh eligible cases for the specified cohort.")

    case_ids = [str(case["case_id"]) for case in selected]
    split_rng = random.Random(int(config["cohort"]["seed"]) + 1)
    shuffled_ids = list(case_ids)
    split_rng.shuffle(shuffled_ids)
    development_ids = sorted(shuffled_ids[: int(config["cohort"]["development_case_count"])])
    confirmation_ids = sorted(shuffled_ids[int(config["cohort"]["development_case_count"]):])
    if len(confirmation_ids) != int(config["cohort"]["confirmation_case_count"]):
        raise RuntimeError("Fresh cohort split does not match the specified counts.")

    questions: list[dict] = []
    chunks: list[dict] = []
    for case in selected:
        case_chunks = build_case_chunks(case)
        chunks.extend(case_chunks)
        questions.extend(build_case_questions(case, case_chunks))

    development_set = set(development_ids)
    confirmation_set = set(confirmation_ids)
    payload = {
        "benchmark": "OpenI fresh multimodal V5 cohort",
        "version": "5.0",
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_cases": portable_path(args.cases),
        "excluded_case_count": len(excluded),
        "excluded_case_fingerprint_sha256": fingerprint(sorted(excluded)),
        "case_count": len(selected),
        "case_ids": case_ids,
        "case_id_fingerprint_sha256": fingerprint(case_ids),
        "question_count": len(questions),
        "chunk_count": len(chunks),
        "questions": questions,
        "chunks": chunks,
        "split": {
            "development": {
                "case_ids": development_ids,
                "qids": [row["qid"] for row in questions if row["case_id"] in development_set],
            },
            "confirmation": {
                "case_ids": confirmation_ids,
                "qids": [row["qid"] for row in questions if row["case_id"] in confirmation_set],
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "excluded_case_count": len(excluded),
        "case_count": len(selected),
        "question_count": len(questions),
        "case_id_fingerprint_sha256": payload["case_id_fingerprint_sha256"],
        "development_cases": len(development_ids),
        "confirmation_cases": len(confirmation_ids),
    }, indent=2))


if __name__ == "__main__":
    main()
