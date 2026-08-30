"""Select the frozen Calibration-only MedSigLIP history-confidence gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_final_qa_history_policy import negative_transfer
from evaluate_final_qa_qlora_pilot import bootstrap_difference, keyed, metrics, read_jsonl
from run_final_qa_report_text_rag_pilot import _embedding_map

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.similar_case.radgraph_adapter import read_radgraph_case_records  # noqa: E402


B3 = "b3_no_history_r2"
B6 = "b6_top1_image_neighbor_whole_report"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def top1_scores(args: argparse.Namespace, target_ids: set[str]) -> dict[str, float]:
    manifest = load_json(args.manifest)
    raw_cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    embeddings, _ = _embedding_map(args.embeddings)
    radgraph = read_radgraph_case_records(args.radgraph)
    train_cases = manifest["roles"]["train"]["cases"]
    bank_clusters = {str(case["case_id"]): str(case["cluster_id"]) for case in train_cases}
    bank_ids = sorted(
        case_id
        for case_id in bank_clusters
        if case_id in raw_cases
        and case_id in embeddings
        and case_id in radgraph
        and radgraph[case_id].status == "ok"
        and (raw_cases[case_id].get("findings") or raw_cases[case_id].get("impression"))
    )
    bank_matrix = np.stack([embeddings[case_id] for case_id in bank_ids])
    target_clusters = {
        str(case["case_id"]): str(case["cluster_id"])
        for case in manifest["roles"]["calibration"]["cases"]
    }
    result: dict[str, float] = {}
    for target_id in sorted(target_ids):
        scores = embeddings[target_id] @ bank_matrix.T
        eligible = np.asarray(
            [
                bank_clusters[case_id] != target_clusters[target_id]
                and case_id != target_id
                for case_id in bank_ids
            ],
            dtype=bool,
        )
        scores[~eligible] = -np.inf
        result[target_id] = float(np.max(scores))
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config)
    rows = read_jsonl(args.rows)
    b3 = keyed(rows, B3)
    b6 = keyed(rows, B6)
    if set(b3) != set(b6):
        raise RuntimeError("B3 and B6 do not use identical Calibration rows")
    scores = top1_scores(args, {key[0] for key in b3})
    values = np.asarray([scores[case_id] for case_id in sorted(scores)], dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    selected_rows_by_quantile: dict[float, dict[tuple[str, int], dict[str, Any]]] = {}
    for quantile in [float(value) for value in config["quantile_grid"]]:
        threshold = float(np.quantile(values, quantile))
        selected = {
            key: (b6[key] if scores[key[0]] >= threshold else b3[key])
            for key in b3
        }
        selected_rows_by_quantile[quantile] = selected
        selected_metrics = metrics(selected.values())
        candidates.append(
            {
                "quantile": quantile,
                "threshold": threshold,
                "history_case_coverage": float(
                    np.mean([scores[case_id] >= threshold for case_id in scores])
                ),
                "metrics": selected_metrics,
                "negative_transfer_from_b3": negative_transfer(b3, selected),
            }
        )
    selected_candidate = sorted(
        candidates,
        key=lambda row: (-float(row["metrics"]["option_micro_f1"]), float(row["quantile"])),
    )[0]
    selected_rows = selected_rows_by_quantile[float(selected_candidate["quantile"])]
    ungated_metrics = metrics(b6.values())
    micro_gain = float(selected_candidate["metrics"]["option_micro_f1"]) - float(
        ungated_metrics["option_micro_f1"]
    )
    ungated_negative = negative_transfer(b3, b6)
    selected_negative = selected_candidate["negative_transfer_from_b3"]
    advanced = (
        micro_gain >= float(config["advancement"]["minimum_micro_f1_gain_over_ungated_b6"])
        and float(selected_negative["rate"]) <= float(ungated_negative["rate"])
    )
    summary = {
        "study": config["study"],
        "score_distribution": {
            "case_count": len(values),
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "maximum": float(values.max()),
        },
        "candidates": candidates,
        "ungated_b6": {
            "metrics": ungated_metrics,
            "negative_transfer_from_b3": ungated_negative,
        },
        "selected_gate": selected_candidate,
        "selected_minus_ungated_micro_f1": micro_gain,
        "case_grouped_bootstrap_selected_vs_ungated": bootstrap_difference(
            selected_rows, b6
        ),
        "case_grouped_bootstrap_selected_vs_b3": bootstrap_difference(
            selected_rows, b3
        ),
        "prespecified_gate_advancement_rule_passed": advanced,
        "final_calibration_policy": "selective_gate" if advanced else "ungated_b6",
        "boundary": config["boundary"],
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/final_qa_selective_history_gate.json",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "experiments/final_qa_development/final_qa_qlora_384_calibration_rows.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json",
    )
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument(
        "--embeddings", type=Path, default=ROOT / "data/processed/v10_medsiglip_embeddings.npz"
    )
    parser.add_argument(
        "--radgraph", type=Path, default=ROOT / "data/processed/v9_radgraph_modern_xl.jsonl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/final_qa_development/final_qa_selective_history_gate.json",
    )
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
