"""Build leakage-controlled V16 multimodal SFT examples.

The builder uses only the V10 Train partition for fitting examples. Historical
cases are selected by a fixed BM25/image RRF rule from the Train bank, with the
target case and its duplicate cluster removed. The generated answers come from
the target case's report section and are supervision only; they are never part
of the inference prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.multimodal.v9_generation import select_primary_image  # noqa: E402
from medical_rag.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from medical_rag.retrieval.candidate_generation import reciprocal_rank_fusion_union  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402


QUESTIONS = {
    "findings": "What are the main radiographic findings?",
    "impression": "What is the most likely radiographic impression?",
}
CONDITIONS = ("no_history", "retrieved_history", "random_history")
SPLIT_PARTITION = "train"
SELECTION_SEED = 1617
RRF_SOURCE_TOP_K = 100
RRF_OUTPUT_K = 200
HISTORY_TOP_K = 3
MAX_SECTION_CHARS = 900


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def digest_key(*parts: str) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_rank(values: Sequence[str], *parts: str) -> list[str]:
    return sorted(
        (str(value) for value in values),
        key=lambda value: (digest_key(*parts, value), value),
    )


def truncate_sentences(value: Any, *, maximum_chars: int = MAX_SECTION_CHARS) -> str:
    text = canonical_text(value)
    if len(text) <= maximum_chars:
        return text
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    output = ""
    for sentence in sentences:
        candidate = sentence if not output else f"{output} {sentence}"
        if len(candidate) > maximum_chars:
            break
        output = candidate
    return output or text[:maximum_chars].rstrip()


def bounded_answer(value: Any, *, maximum_sentences: int = 2, maximum_chars: int = 700) -> str:
    text = canonical_text(value)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    bounded = " ".join(sentences[:maximum_sentences]) if sentences else text
    return bounded[:maximum_chars].rstrip()


def build_prompt(
    *,
    indication: Any,
    question: str,
    question_type: str,
    retrieved_cases: Sequence[Mapping[str, Any]],
) -> str:
    if retrieved_cases:
        evidence: list[str] = []
        for index, case in enumerate(retrieved_cases, start=1):
            evidence.extend(
                [
                    f"Historical case {index} (other patient):",
                    f"Findings: {truncate_sentences(case.get('findings')) or 'Not documented'}",
                    f"Impression: {truncate_sentences(case.get('impression')) or 'Not documented'}",
                ]
            )
    else:
        evidence = ["No historical cases were provided."]
    history_rule = (
        "No historical evidence is available. Answer from the target image and indication only."
        if not retrieved_cases
        else "Historical reports describe other patients and are analogy only, not proof about the target patient."
    )
    return "\n".join(
        [
            "You are a cautious radiology question-answering assistant.",
            "Use the target chest radiograph as the primary patient evidence.",
            history_rule,
            "Do not transfer a historical finding to the target patient unless it is supported by the target image.",
            "Answer the question directly in at most two concise complete sentences.",
            "Do not output JSON, labels, citations, analysis, or a preamble.",
            f"Question type: {question_type}",
            f"Clinical indication: {canonical_text(indication) or 'Not provided'}",
            f"Question: {canonical_text(question)}",
            "Historical evidence:",
            *evidence,
        ]
    )


def make_ranking(
    *,
    target_id: str,
    excluded_ids: set[str],
    bank_ids: Sequence[str],
    bm25_scores: Sequence[float],
    image_scores: Sequence[float],
) -> list[str]:
    text_rank = [
        case_id
        for _, case_id in sorted(
            ((float(score), case_id) for score, case_id in zip(bm25_scores, bank_ids, strict=True)),
            key=lambda row: (-row[0], row[1]),
        )
        if case_id not in excluded_ids and case_id != target_id
    ]
    image_rank = [
        case_id
        for _, case_id in sorted(
            ((float(score), case_id) for score, case_id in zip(image_scores, bank_ids, strict=True)),
            key=lambda row: (-row[0], row[1]),
        )
        if case_id not in excluded_ids and case_id != target_id
    ]
    return reciprocal_rank_fusion_union(
        [text_rank, image_rank],
        source_top_k=RRF_SOURCE_TOP_K,
        output_k=RRF_OUTPUT_K,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/v10/v10_cluster_disjoint_split.json")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/processed/v10_medsiglip_embeddings.npz")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/openi_official_images")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v16_adaptation/v16_sft_examples.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "data/splits/v16/v16_sft_dataset_summary.json")
    parser.add_argument("--limit-cases", type=int, default=None)
    args = parser.parse_args()

    raw_cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    split = read_json(args.split)
    train_ids = [str(value) for value in split["partitions"][SPLIT_PARTITION]["case_ids"]]
    clusters: dict[str, str] = {}
    for cluster in split.get("clusters", []):
        cluster_id = str(cluster["cluster_id"])
        for case_id in cluster["case_ids"]:
            clusters[str(case_id)] = cluster_id

    with np.load(args.embeddings, allow_pickle=False) as encoded:
        embedding_ids = [str(value) for value in encoded["case_ids"].tolist()]
        image_embeddings = np.asarray(encoded["case_image_embeddings"], dtype=np.float32)
    image_by_id = {case_id: image_embeddings[index] for index, case_id in enumerate(embedding_ids)}
    bank_ids = [case_id for case_id in train_ids if case_id in raw_cases and case_id in image_by_id]
    if len(bank_ids) < 100:
        raise RuntimeError(f"V16 historical bank is unexpectedly small: {len(bank_ids)}")
    bank_cases = [
        {"case_id": case_id, "report_text": raw_cases[case_id].get("report_text", "")}
        for case_id in bank_ids
    ]
    bm25 = BM25Retriever().fit(bank_cases)
    bank_matrix = np.stack([image_by_id[case_id] for case_id in bank_ids])
    eligible_targets = [case_id for case_id in bank_ids if clusters.get(case_id)]
    if args.limit_cases is not None:
        eligible_targets = stable_rank(eligible_targets, "v16-target-limit", str(SELECTION_SEED))[: args.limit_cases]
    eligible_targets = sorted(eligible_targets)

    examples: list[dict[str, Any]] = []
    skipped_pairs: list[dict[str, str]] = []
    available_counts = {question_type: 0 for question_type in QUESTIONS}
    history_counts = {condition: 0 for condition in CONDITIONS}
    for position, case_id in enumerate(eligible_targets, start=1):
        source = raw_cases[case_id]
        excluded_ids = {
            candidate_id
            for candidate_id in bank_ids
            if clusters.get(candidate_id) == clusters.get(case_id)
        }
        for question_type, question in QUESTIONS.items():
            answer = bounded_answer(source.get(question_type, ""))
            if not answer:
                skipped_pairs.append(
                    {
                        "case_id": case_id,
                        "question_type": question_type,
                        "reason": "empty_target_report_section",
                    }
                )
                continue
            available_counts[question_type] += 1
            query_text = "\n".join(
                part for part in (canonical_text(source.get("indication")), question) if part
            )
            bm25_scores = bm25.score_all(query_text)
            image_scores = bank_matrix @ image_by_id[case_id]
            ranking = make_ranking(
                target_id=case_id,
                excluded_ids=excluded_ids,
                bank_ids=bank_ids,
                bm25_scores=bm25_scores,
                image_scores=image_scores,
            )
            random_ids = stable_rank(
                [candidate_id for candidate_id in bank_ids if candidate_id not in excluded_ids and candidate_id != case_id],
                "v16-random-history",
                str(SELECTION_SEED),
                case_id,
                question_type,
            )[:HISTORY_TOP_K]
            history_by_condition = {
                "no_history": [],
                "retrieved_history": [raw_cases[value] for value in ranking[:HISTORY_TOP_K]],
                "random_history": [raw_cases[value] for value in random_ids],
            }
            for condition in CONDITIONS:
                histories = history_by_condition[condition]
                prompt = build_prompt(
                    indication=source.get("indication", ""),
                    question=question,
                    question_type=question_type,
                    retrieved_cases=histories,
                )
                examples.append(
                    {
                        "case_id": case_id,
                        "cluster_id": clusters[case_id],
                        "question_type": question_type,
                        "condition": condition,
                        "image_path": str(select_primary_image(source, args.image_root)),
                        "prompt": prompt,
                        "answer": answer,
                        "retrieved_case_ids": [str(value["case_id"]) for value in histories],
                        "selection_rule": "BM25+MedSigLIP image RRF Top-200; target cluster excluded",
                        "selection_seed": SELECTION_SEED,
                    }
                )
                history_counts[condition] += len(histories)
        if position % 100 == 0 or position == len(eligible_targets):
            print(f"built_cases={position}/{len(eligible_targets)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(example, ensure_ascii=True, separators=(",", ":")) + "\n" for example in examples),
        encoding="utf-8",
    )
    summary = {
        "study": "V16 leakage-controlled multimodal SFT dataset",
        "status": "train_only_examples",
        "source_cases_sha256": file_sha256(args.cases),
        "split_sha256": file_sha256(args.split),
        "embeddings_sha256": file_sha256(args.embeddings),
        "source_partition": SPLIT_PARTITION,
        "case_count": len(eligible_targets),
        "question_types": list(QUESTIONS),
        "conditions": list(CONDITIONS),
        "example_count": len(examples),
        "available_target_sections_by_question": available_counts,
        "skipped_pair_count": len(skipped_pairs),
        "skipped_pairs_sha256": hashlib.sha256(
            "\n".join(
                f"{row['case_id']}|{row['question_type']}|{row['reason']}"
                for row in sorted(skipped_pairs, key=lambda value: (value["case_id"], value["question_type"]))
            ).encode("utf-8")
        ).hexdigest(),
        "examples_by_condition": {
            condition: sum(example["condition"] == condition for example in examples)
            for condition in CONDITIONS
        },
        "historical_units_by_condition": history_counts,
        "selection_seed": SELECTION_SEED,
        "rrf_source_top_k": RRF_SOURCE_TOP_K,
        "rrf_output_k": RRF_OUTPUT_K,
        "history_top_k": HISTORY_TOP_K,
        "case_ids_sha256": hashlib.sha256(
            "\n".join(eligible_targets).encode("utf-8")
        ).hexdigest(),
        "output_sha256": file_sha256(args.output),
        "claim_boundary": "Train supervision only; no clinical labels or independent patient validation.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
