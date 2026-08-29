"""Evaluate paired V16 outputs with standard language-generation metrics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge


CONDITIONS = ("no_history", "retrieved_history", "random_history")
QUESTION_TYPES = ("findings", "impression")
LINEAR_METRICS = ("bleu_1", "bleu_4", "rouge_l", "meteor", "cider", "bertscore_f1")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["case_id"]), str(row["question_type"]), str(row["condition"])


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def validate(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    keys = [row_key(row) for row in rows]
    if not keys or len(keys) != len(set(keys)):
        raise RuntimeError(f"Empty or duplicated {label} rows")
    if {key[1] for key in keys} != set(QUESTION_TYPES):
        raise RuntimeError(f"Unexpected {label} question matrix")
    if {key[2] for key in keys} != set(CONDITIONS):
        raise RuntimeError(f"Unexpected {label} condition matrix")


def lexical_scores(references: Sequence[str], predictions: Sequence[str]) -> tuple[dict[str, float], list[dict[str, float]]]:
    if len(references) != len(predictions) or not references:
        raise RuntimeError("References and predictions must be non-empty and paired")
    gts = {index: [reference] for index, reference in enumerate(references)}
    res = {index: [prediction] for index, prediction in enumerate(predictions)}
    bleu_corpus, bleu_rows = Bleu(4).compute_score(gts, res)
    rouge_corpus, rouge_rows = Rouge().compute_score(gts, res)
    cider_corpus, cider_rows = Cider().compute_score(gts, res)
    rows = []
    for index, (reference, prediction) in enumerate(zip(references, predictions, strict=True)):
        rows.append({
            "bleu_1": float(bleu_rows[0][index]),
            "bleu_4": float(bleu_rows[3][index]),
            "rouge_l": float(rouge_rows[index]),
            "meteor": float(meteor_score([reference.split()], prediction.split())),
            "cider": float(cider_rows[index]),
        })
    corpus = {
        "bleu_1": float(bleu_corpus[0]),
        "bleu_4": float(bleu_corpus[3]),
        "rouge_l": float(rouge_corpus),
        "cider": float(cider_corpus),
        "meteor_row_mean": mean([row["meteor"] for row in rows]),
    }
    return corpus, rows


def add_bertscore(
    rows: list[dict[str, float]],
    references: Sequence[str],
    predictions: Sequence[str],
    *,
    model_type: str,
    batch_size: int,
    device: str,
) -> str:
    from bert_score import score

    _, _, f1, model_hash = score(
        list(predictions),
        list(references),
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        rescale_with_baseline=True,
        return_hash=True,
        verbose=False,
    )
    for row, value in zip(rows, f1.cpu().numpy().tolist(), strict=True):
        row["bertscore_f1"] = float(value)
    return str(model_hash)


def paired_case_bootstrap(
    left: Mapping[tuple[str, str, str], Mapping[str, float]],
    right: Mapping[tuple[str, str, str], Mapping[str, float]],
    metric: str,
    condition: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if set(left) != set(right):
        raise RuntimeError("Paired score keys differ")
    by_case: dict[str, list[float]] = defaultdict(list)
    for key in sorted(left):
        if key[2] == condition:
            by_case[key[0]].append(float(left[key][metric]) - float(right[key][metric]))
    values = np.asarray([mean(by_case[case_id]) for case_id in sorted(by_case)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975]).tolist()
    return {
        "case_count": len(values),
        "mean_difference": float(values.mean()),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "iterations": iterations,
        "seed": seed,
    }


def run(args: argparse.Namespace) -> None:
    arms = {
        args.left_label: read_jsonl(args.left_rows),
        args.right_label: read_jsonl(args.right_rows),
    }
    for label, rows in arms.items():
        validate(rows, label)
    if {row_key(row) for row in arms[args.left_label]} != {row_key(row) for row in arms[args.right_label]}:
        raise RuntimeError("Paired generation matrices differ")

    row_scores: dict[str, dict[tuple[str, str, str], dict[str, float]]] = {}
    summaries: dict[str, Any] = {}
    bert_hashes: dict[str, str] = {}
    for label, rows in arms.items():
        references = [str(row.get("reference_answer") or "") for row in rows]
        predictions = [str(row.get("answer") or "") for row in rows]
        corpus, scores = lexical_scores(references, predictions)
        if args.skip_bertscore:
            for score_row in scores:
                score_row["bertscore_f1"] = 0.0
            bert_hashes[label] = "skipped"
        else:
            bert_hashes[label] = add_bertscore(
                scores,
                references,
                predictions,
                model_type=args.bertscore_model,
                batch_size=args.bertscore_batch_size,
                device=args.device,
            )
        row_scores[label] = {row_key(row): score_row for row, score_row in zip(rows, scores, strict=True)}
        corpus_by_condition = {}
        for condition in CONDITIONS:
            selected = [row for row in rows if str(row["condition"]) == condition]
            condition_corpus, _ = lexical_scores(
                [str(row.get("reference_answer") or "") for row in selected],
                [str(row.get("answer") or "") for row in selected],
            )
            corpus_by_condition[condition] = condition_corpus
        summaries[label] = {
            "corpus": corpus,
            "corpus_by_condition": corpus_by_condition,
            "row_means": {metric: mean([score_row[metric] for score_row in scores]) for metric in LINEAR_METRICS},
            "by_condition": {
                condition: {
                    metric: mean([
                        row_scores[label][key][metric]
                        for key in row_scores[label]
                        if key[2] == condition
                    ])
                    for metric in LINEAR_METRICS
                }
                for condition in CONDITIONS
            },
            "by_question_type": {
                question_type: {
                    metric: mean([
                        row_scores[label][key][metric]
                        for key in row_scores[label]
                        if key[1] == question_type
                    ])
                    for metric in LINEAR_METRICS
                }
                for question_type in QUESTION_TYPES
            },
        }

    comparisons = {
        condition: {
            metric: paired_case_bootstrap(
                row_scores[args.left_label],
                row_scores[args.right_label],
                metric,
                condition,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed + condition_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(LINEAR_METRICS)
        }
        for condition_index, condition in enumerate(CONDITIONS)
    }
    output = {
        "study": "V16 standard NLG confirmation evaluation",
        "status": "confirmation_evaluation_no_retuning",
        "counts": {"cases": len({key[0] for key in row_scores[args.left_label]}), "rows_per_arm": len(arms[args.left_label])},
        "arms": summaries,
        f"{args.left_label}_minus_{args.right_label}": comparisons,
        "implementation": {
            "pycocoevalcap": importlib.metadata.version("pycocoevalcap"),
            "nltk": importlib.metadata.version("nltk"),
            "rouge_score": importlib.metadata.version("rouge-score"),
            "bert_score": importlib.metadata.version("bert-score"),
            "bertscore_model": args.bertscore_model,
            "bertscore_hashes": bert_hashes,
            "bertscore_rescale_with_baseline": True,
            "meteor": "nltk.translate.meteor_score with WordNet",
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "claim_boundary": "Automated same-source report-reference consistency; not diagnostic accuracy or clinical safety.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-rows", type=Path, required=True)
    parser.add_argument("--right-rows", type=Path, required=True)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1626)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
