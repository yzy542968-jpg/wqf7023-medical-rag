from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from textwrap import shorten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.answer_metrics import extract_final_answer, token_f1


SYSTEMS = [
    (
        "llm_only",
        "LLM-only Qwen2.5-1.5B",
        Path("experiments/generations_llm_only_qwen15_full360.jsonl"),
    ),
    (
        "report_bm25",
        "Report-RAG BM25 Qwen2.5-1.5B + checker",
        Path("experiments/generations_report_rag_bm25_qwen15_full360_agentic_top1.jsonl"),
    ),
    (
        "case_bm25_top1",
        "Case-RAG BM25 top-1 Qwen2.5-1.5B + checker",
        Path("experiments/generations_case_rag_bm25_top1_qwen15_full360_agentic_top1.jsonl"),
    ),
    (
        "case_hybrid_top1",
        "Case-RAG Hybrid top-1 Qwen2.5-1.5B + checker",
        Path("experiments/generations_case_rag_hybrid_top1_qwen15_full360_agentic_top1.jsonl"),
    ),
]


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _agent(row: dict) -> dict:
    return row.get("agent", {})


def _support_rate(row: dict) -> float | None:
    value = _agent(row).get("support_rate")
    return float(value) if value is not None else None


def _revision(row: dict) -> bool | None:
    value = _agent(row).get("revised")
    return bool(value) if value is not None else None


def _abstained(row: dict) -> bool | None:
    value = _agent(row).get("abstained")
    return bool(value) if value is not None else None


def _unsupported_count(row: dict) -> int:
    return len(_agent(row).get("unsupported_sentences", []))


def _answer(row: dict) -> str:
    return row.get("answer", "") if "agent" in row else extract_final_answer(row.get("answer", ""))


def _draft_answer(row: dict) -> str:
    return row.get("source_generation_answer") or extract_final_answer(row.get("answer", ""))


def _top1_hit(row: dict) -> bool:
    retrieved = row.get("retrieved_case_ids", [])
    relevant = set(row.get("relevant_case_ids", []))
    return bool(retrieved and retrieved[0] in relevant)


def _retrieved_hit(row: dict) -> bool:
    retrieved = set(row.get("retrieved_case_ids", []))
    relevant = set(row.get("relevant_case_ids", []))
    return bool(retrieved.intersection(relevant))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(values: list[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def _summarize(rows: list[dict]) -> dict:
    f1s = [row["answer_token_f1"] for row in rows]
    summary = {
        "n": len(rows),
        "token_f1": _mean(f1s),
        "median_token_f1": statistics.median(f1s) if f1s else 0.0,
        "top1_case_accuracy": _rate([row["top1_hit"] for row in rows]),
        "retrieved_case_hit_rate": _rate([row["retrieved_hit"] for row in rows]),
        "avg_words": _mean([row["answer_words"] for row in rows]),
    }
    support_rows = [row for row in rows if row["evidence_support_rate"] is not None]
    if support_rows:
        summary.update(
            {
                "evidence_support_rate": _mean([row["evidence_support_rate"] for row in support_rows]),
                "revision_rate": _rate([row["revised"] for row in support_rows]),
                "abstention_rate": _rate([row["abstained"] for row in support_rows]),
                "unsupported_sentence_rate": sum(row["unsupported_sentence_count"] for row in support_rows)
                / max(sum(row["sentence_count"] for row in support_rows), 1),
            }
        )
    return summary


def _prepare_rows(system_key: str, system_label: str, records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        answer = _answer(record)
        sentence_checks = _agent(record).get("sentence_checks", [])
        sentence_count = len(sentence_checks) if sentence_checks else 0
        rows.append(
            {
                "system_key": system_key,
                "system_label": system_label,
                "qid": record.get("qid", ""),
                "case_id": record.get("case_id", ""),
                "question_type": record.get("question_type", ""),
                "question": record.get("question", ""),
                "reference_answer": record.get("reference_answer", ""),
                "answer": answer,
                "draft_answer": _draft_answer(record),
                "retrieved_case_ids": record.get("retrieved_case_ids", []),
                "top1_hit": _top1_hit(record),
                "retrieved_hit": _retrieved_hit(record),
                "answer_token_f1": token_f1(answer, record.get("reference_answer", "")),
                "draft_token_f1": token_f1(_draft_answer(record), record.get("reference_answer", "")),
                "answer_words": len(answer.split()),
                "evidence_support_rate": _support_rate(record),
                "revised": _revision(record),
                "abstained": _abstained(record),
                "unsupported_sentence_count": _unsupported_count(record),
                "sentence_count": sentence_count,
                "unsupported_sentences": _agent(record).get("unsupported_sentences", []),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "system_key",
        "system_label",
        "qid",
        "case_id",
        "question_type",
        "top1_hit",
        "retrieved_hit",
        "answer_token_f1",
        "draft_token_f1",
        "answer_words",
        "evidence_support_rate",
        "revised",
        "abstained",
        "unsupported_sentence_count",
        "sentence_count",
        "question",
        "reference_answer",
        "answer",
        "draft_answer",
        "retrieved_case_ids",
        "unsupported_sentences",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = row.copy()
            output_row["retrieved_case_ids"] = " | ".join(row["retrieved_case_ids"])
            output_row["unsupported_sentences"] = " | ".join(row["unsupported_sentences"])
            writer.writerow(output_row)


def _format_float(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _example_block(title: str, row: dict) -> list[str]:
    return [
        f"### {title}",
        "",
        f"- System: {row['system_label']}",
        f"- QID: `{row['qid']}`",
        f"- Question type: `{row['question_type']}`",
        f"- Top-1 hit: {row['top1_hit']}; retrieved hit: {row['retrieved_hit']}",
        f"- Token-F1: {_format_float(row['answer_token_f1'])}",
        f"- Evidence support: {_format_float(row['evidence_support_rate'])}",
        f"- Revised: {row['revised']}; abstained: {row['abstained']}",
        "",
        f"Question: {row['question']}",
        "",
        f"Reference: {shorten(row['reference_answer'], width=500, placeholder='...')}",
        "",
        f"Answer: {shorten(row['answer'], width=700, placeholder='...')}",
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze full360 generation and evidence-checking results.")
    parser.add_argument("--output-dir", default=Path("experiments/full360_analysis"), type=Path)
    args = parser.parse_args()

    all_rows: list[dict] = []
    for system_key, system_label, path in SYSTEMS:
        all_rows.extend(_prepare_rows(system_key, system_label, _load_jsonl(path)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "full360_per_answer_metrics.csv", all_rows)

    by_system = {}
    for system_key, _, _ in SYSTEMS:
        rows = [row for row in all_rows if row["system_key"] == system_key]
        by_system[system_key] = _summarize(rows)

    by_question_type: dict[tuple[str, str], dict] = {}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["system_key"], row["question_type"])].append(row)
    for key, rows in grouped.items():
        by_question_type[key] = _summarize(rows)

    json_payload = {
        "by_system": by_system,
        "by_question_type": {f"{system}|{qtype}": summary for (system, qtype), summary in by_question_type.items()},
    }
    (args.output_dir / "full360_grouped_metrics.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    system_rows = []
    for system_key, system_label, _ in SYSTEMS:
        summary = by_system[system_key]
        system_rows.append(
            [
                system_label,
                str(summary["n"]),
                _format_float(summary["token_f1"]),
                _format_float(summary["top1_case_accuracy"]),
                _format_float(summary["retrieved_case_hit_rate"]),
                _format_float(summary.get("evidence_support_rate")),
                _format_float(summary.get("revision_rate")),
                _format_float(summary.get("abstention_rate")),
                _format_float(summary.get("unsupported_sentence_rate")),
            ]
        )

    qtype_rows = []
    for (system_key, qtype), summary in sorted(by_question_type.items()):
        qtype_rows.append(
            [
                system_key,
                qtype,
                str(summary["n"]),
                _format_float(summary["token_f1"]),
                _format_float(summary["retrieved_case_hit_rate"]),
                _format_float(summary.get("evidence_support_rate")),
                _format_float(summary.get("abstention_rate")),
            ]
        )

    candidate_rows = [row for row in all_rows if row["system_key"] != "llm_only"]
    low_support_rows = [
        row
        for row in candidate_rows
        if row["evidence_support_rate"] is not None and row["evidence_support_rate"] <= 0.35
    ]
    examples = [
        ("High-support success case", max(candidate_rows, key=lambda row: (row["evidence_support_rate"] or 0, row["answer_token_f1"]))),
        ("High-F1 but low-support case", max(low_support_rows, key=lambda row: row["answer_token_f1"])),
        ("Retrieval miss with poor answer", min([row for row in candidate_rows if not row["retrieved_hit"]], key=lambda row: row["answer_token_f1"])),
        ("Heavy revision case", max(candidate_rows, key=lambda row: row["unsupported_sentence_count"])),
        ("Hybrid top-1 representative failure", min([row for row in all_rows if row["system_key"] == "case_hybrid_top1"], key=lambda row: row["answer_token_f1"])),
    ]

    md_lines = [
        "# Full360 Automated Error Analysis",
        "",
        "This report is generated from the full 360-question Qwen2.5-1.5B outputs. It is an automatic analysis and should be followed by manual annotation.",
        "",
        "## System-Level Summary",
        "",
        *_md_table(
            [
                "System",
                "N",
                "Token-F1",
                "Top-1 Hit",
                "Retrieved Hit",
                "Evidence Support",
                "Revision",
                "Abstention",
                "Unsupported Sentence Rate",
            ],
            system_rows,
        ),
        "",
        "## Question-Type Summary",
        "",
        *_md_table(
            ["System", "Question Type", "N", "Token-F1", "Retrieved Hit", "Evidence Support", "Abstention"],
            qtype_rows,
        ),
        "",
        "## Representative Cases",
        "",
    ]
    for title, row in examples:
        md_lines.extend(_example_block(title, row))

    (args.output_dir / "full360_error_analysis.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "rows": len(all_rows),
                "output_dir": str(args.output_dir),
                "metrics_csv": str(args.output_dir / "full360_per_answer_metrics.csv"),
                "grouped_metrics": str(args.output_dir / "full360_grouped_metrics.json"),
                "markdown": str(args.output_dir / "full360_error_analysis.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
