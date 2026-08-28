"""Evaluate paired V15 default/deeper retrieval generation with frozen metrics."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_v10_pathology_utility import (  # noqa: E402
    label_unique_texts,
    resolve_checkpoint,
    text_sha256,
)
from evaluate_v13_concept_qa_pilot import (  # noqa: E402
    RADGRAPH_METRICS,
    grouped_values,
    paired_bootstrap,
    score_radgraph,
)
from medical_rag.evaluation.chexbert_pathology import (  # noqa: E402
    METRIC_NAMES as CHEXBERT_METRICS,
    build_case_statistics,
    metrics_from_case_statistics,
    paired_case_bootstrap,
)
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from run_v10_evidence_generation_development import read_json, read_jsonl  # noqa: E402

CONDITIONS = ("default_17", "deeper_17")


def combined_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    deeper_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    baseline = [
        {
            **dict(row),
            "condition": "default_17",
        }
        for row in baseline_rows
        if row["policy"] == "whole_report" and int(row["max_new_tokens"]) == 96
    ]
    deeper = [{**dict(row), "condition": "deeper_17"} for row in deeper_rows]
    keys = {
        condition: {
            (str(row["case_id"]), str(row["question_type"]))
            for row in baseline + deeper
            if row["condition"] == condition
        }
        for condition in CONDITIONS
    }
    if keys["default_17"] != keys["deeper_17"] or len(keys["default_17"]) != 144:
        raise RuntimeError("V15 condition matrices differ or are incomplete")
    return baseline + deeper


def select_scope(rows: Sequence[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    if scope == "all":
        return list(rows)
    if scope == "primary":
        return [row for row in rows if not bool(row["reference_is_proxy"])]
    return [row for row in rows if str(row["question_type"]) == scope]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-rows", type=Path, default=ROOT / "experiments/v12_optimization/generation/v12_generation_96_rows.jsonl")
    parser.add_argument("--deeper-rows", type=Path, default=ROOT / "experiments/v15_retrieval_transfer/v15_deeper_rows.jsonl")
    parser.add_argument("--generation-summary", type=Path, default=ROOT / "data/splits/v15/v15_retrieval_transfer_generation_summary.json")
    parser.add_argument("--chexbert-cache", type=Path, default=ROOT / "experiments/v15_retrieval_transfer/v15_chexbert_cache.json")
    parser.add_argument("--radgraph-output", type=Path, default=ROOT / "experiments/v15_retrieval_transfer/v15_radgraph.csv")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "data/splits/v15/v15_retrieval_transfer_evaluation_summary.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chexbert-batch-size", type=int, default=128)
    parser.add_argument("--radgraph-model", default="modern-radgraph-xl")
    parser.add_argument("--radgraph-batch-size", type=int, default=8)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1516)
    args = parser.parse_args()

    generation_summary = read_json(args.generation_summary)
    if generation_summary.get("status") != "generation_complete_metric_extension_pending":
        raise RuntimeError("V15 generation summary is not ready for metric evaluation")
    if generation_summary["inputs"]["baseline_rows_sha256"] != file_sha256(args.baseline_rows):
        raise RuntimeError("V15 baseline row hash differs")
    if generation_summary["artifacts"]["deeper_rows_sha256"] != file_sha256(args.deeper_rows):
        raise RuntimeError("V15 deeper row hash differs")
    rows = combined_rows(read_jsonl(args.baseline_rows), read_jsonl(args.deeper_rows))

    checkpoint = resolve_checkpoint()
    labels_by_hash = label_unique_texts(
        rows,
        cache_path=args.chexbert_cache,
        checkpoint_hash=file_sha256(checkpoint),
        device=args.device,
        batch_size=args.chexbert_batch_size,
    )
    radgraph_by_key = score_radgraph(
        rows,
        output_path=args.radgraph_output,
        model_type=args.radgraph_model,
        batch_size=args.radgraph_batch_size,
        cuda=args.cuda,
    )
    enriched = []
    for source in rows:
        row = dict(source)
        key = (str(row["case_id"]), str(row["question_type"]), str(row["condition"]))
        row.update(radgraph_by_key[key])
        row["reference_labels"] = np.asarray(
            labels_by_hash[text_sha256(str(row.get("reference_answer") or ""))],
            dtype=np.int8,
        )
        row["prediction_labels"] = np.asarray(
            labels_by_hash[text_sha256(str(row.get("answer") or ""))],
            dtype=np.int8,
        )
        enriched.append(row)

    scopes = {}
    for scope_index, scope in enumerate(("primary", "all", "findings", "impression", "acute")):
        scoped = select_scope(enriched, scope)
        by_condition = {
            condition: [row for row in scoped if row["condition"] == condition]
            for condition in CONDITIONS
        }
        linear_metrics = ("token_f1", *RADGRAPH_METRICS)
        systems = {
            metric: {
                condition: statistics.fmean(float(row[metric]) for row in by_condition[condition])
                for condition in CONDITIONS
            }
            for metric in linear_metrics
        }
        comparisons = {
            metric: paired_bootstrap(
                grouped_values(by_condition["deeper_17"], metric),
                grouped_values(by_condition["default_17"], metric),
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed + scope_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(linear_metrics)
        }
        chex_stats = {}
        chex_metrics = {}
        for condition in CONDITIONS:
            selected = by_condition[condition]
            stats = build_case_statistics(
                [str(row["case_id"]) for row in selected],
                np.stack([row["reference_labels"] for row in selected]),
                np.stack([row["prediction_labels"] for row in selected]),
            )
            chex_stats[condition] = stats
            chex_metrics[condition] = metrics_from_case_statistics(stats)
        scopes[scope] = {
            "row_count_per_condition": len(by_condition["default_17"]),
            "linear_metrics": systems,
            "deeper_minus_default": comparisons,
            "chexbert_metrics": {
                metric: {
                    condition: float(chex_metrics[condition][metric])
                    for condition in CONDITIONS
                }
                for metric in CHEXBERT_METRICS
            },
            "chexbert_deeper_minus_default": paired_case_bootstrap(
                chex_stats["deeper_17"],
                chex_stats["default_17"],
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed + scope_index * 100 + 20,
            ),
        }

    integrity = {
        condition: {
            "answer_contract_valid_rate": statistics.fmean(
                float(row["answer_only_contract_valid"])
                for row in enriched
                if row["condition"] == condition
            ),
            "provenance_valid_rate": statistics.fmean(
                float(row["evidence_provenance_valid"])
                for row in enriched
                if row["condition"] == condition
            ),
            "token_ceiling_rate": statistics.fmean(
                float(row["hit_token_ceiling"])
                for row in enriched
                if row["condition"] == condition
            ),
        }
        for condition in CONDITIONS
    }
    primary = scopes["primary"]
    promotion = {
        "token_f1_ci_above_zero": primary["deeper_minus_default"]["token_f1"]["ci_95_low"] > 0,
        "complete_radgraph_ci_above_zero": primary["deeper_minus_default"]["f1_radgraph_complete"]["ci_95_low"] > 0,
        "token_f1_not_wholly_negative": primary["deeper_minus_default"]["token_f1"]["ci_95_high"] >= 0,
        "complete_radgraph_not_wholly_negative": primary["deeper_minus_default"]["f1_radgraph_complete"]["ci_95_high"] >= 0,
        "chexbert_micro_f1_5_not_wholly_negative": primary["chexbert_deeper_minus_default"]["micro_f1_5"]["ci_95_high"] >= 0,
        "integrity_not_lower": all(
            integrity["deeper_17"][metric] >= integrity["default_17"][metric]
            for metric in ("answer_contract_valid_rate", "provenance_valid_rate")
        ),
    }
    output = {
        "study": "V15 stronger retrieval to QA transfer evaluation",
        "status": "validation_evaluation_complete_no_retuning",
        "protocol_commit": generation_summary["protocol_commit"],
        "no_test_evaluation": True,
        "counts": {"cases": 48, "rows": len(rows), "conditions": 2},
        "scopes": scopes,
        "integrity": integrity,
        "promotion_diagnostics": promotion,
        "runtime": {
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
            "radgraph_model": args.radgraph_model,
            "f1chexbert_version": "0.0.2",
            "f1chexbert_checkpoint_sha256": file_sha256(checkpoint),
        },
        "artifacts": {
            "baseline_rows_sha256": file_sha256(args.baseline_rows),
            "deeper_rows_sha256": file_sha256(args.deeper_rows),
            "generation_summary_sha256": file_sha256(args.generation_summary),
            "chexbert_cache_sha256": file_sha256(args.chexbert_cache),
            "radgraph_rows_sha256": file_sha256(args.radgraph_output),
            "script_sha256": file_sha256(Path(__file__)),
        },
        "claim_boundary": (
            "Validation-only automated answer-reference consistency; not diagnosis, "
            "clinical correctness, safety, physician utility, or external validation."
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
