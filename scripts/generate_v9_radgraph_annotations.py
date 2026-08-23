from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v6_development_confirmation_separation import (  # noqa: E402
    canonical_case_id,
    case_id_fingerprint,
    file_sha256,
    read_json,
    read_jsonl,
)
from medical_rag.similar_case.radgraph_adapter import (  # noqa: E402
    radgraph_annotation_facts,
)


DEFAULT_CONFIG = ROOT / "config" / "v9_radgraph_preprocessing.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v9" / "v9_full_source_split.json"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "v9_radgraph_modern_xl.jsonl"
DEFAULT_AUDIT = ROOT / "data" / "splits" / "v9" / "v9_radgraph_preprocessing_audit.json"


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def normalized_report_text(row: dict[str, Any]) -> str:
    fields = [" ".join(str(row.get(name, "")).split()) for name in ("findings", "impression")]
    return "\n".join(value for value in fields if value)


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_fact_fingerprint(records: Iterable[dict[str, Any]]) -> str:
    lines: list[str] = []
    for record in sorted(records, key=lambda row: canonical_case_id(row["case_id"])):
        case_id = canonical_case_id(record["case_id"])
        facts = sorted(str(value) for value in record.get("facts", []))
        lines.append(case_id + "\t" + "\u001f".join(facts))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                case_id = canonical_case_id(record["case_id"])
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid checkpoint row {line_number}: {exc}") from exc
            if case_id in records:
                raise ValueError(f"Duplicate checkpoint case ID: {case_id}")
            records[case_id] = record
    return records


def append_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def canonicalize_output(path: Path, records: dict[str, dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for case_id in sorted(records):
            handle.write(
                json.dumps(records[case_id], ensure_ascii=True, sort_keys=True) + "\n"
            )
    os.replace(temporary, path)


def package_version(name: str) -> str:
    from importlib.metadata import version

    return version(name)


def default_model_cache() -> Path:
    from appdirs import user_cache_dir

    return Path(user_cache_dir("radgraph")) / package_version("radgraph")


def split_membership(split: dict[str, Any]) -> dict[str, str]:
    membership: dict[str, str] = {}
    for partition, block in split["partitions"].items():
        for value in block["case_ids"]:
            case_id = canonical_case_id(value)
            if case_id in membership:
                raise ValueError(f"Split manifest repeats case ID {case_id}.")
            membership[case_id] = partition
    return membership


def annotation_record(
    *,
    case_id: str,
    report_text: str,
    model_type: str,
    annotation: dict[str, Any] | None,
    error: str | None = None,
) -> dict[str, Any]:
    if not report_text:
        return {
            "case_id": case_id,
            "report_text_sha256": text_sha256(report_text),
            "model_type": model_type,
            "status": "empty_report",
            "facts": [],
            "annotation": None,
            "error": None,
        }
    if error is not None:
        return {
            "case_id": case_id,
            "report_text_sha256": text_sha256(report_text),
            "model_type": model_type,
            "status": "error",
            "facts": [],
            "annotation": None,
            "error": error,
        }
    if annotation is None:
        raise ValueError("A nonempty successful report requires an annotation.")
    return {
        "case_id": case_id,
        "report_text_sha256": text_sha256(report_text),
        "model_type": model_type,
        "status": "ok",
        "facts": sorted(radgraph_annotation_facts(annotation)),
        "annotation": annotation,
        "error": None,
    }


def build_audit(
    *,
    config_path: Path,
    cases_path: Path,
    split_path: Path,
    output_path: Path,
    records: dict[str, dict[str, Any]],
    model_type: str,
    model_cache_dir: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    config = read_json(config_path)
    split = read_json(split_path)
    membership = split_membership(split)
    status_counts = Counter(record["status"] for record in records.values())
    split_status: dict[str, dict[str, int]] = {}
    for partition in ("train", "validation", "test"):
        partition_records = [
            record
            for case_id, record in records.items()
            if membership.get(case_id) == partition
        ]
        counts = Counter(record["status"] for record in partition_records)
        split_status[partition] = {
            "total": len(partition_records),
            "ok": counts["ok"],
            "empty_report": counts["empty_report"],
            "error": counts["error"],
        }
    strict_ids = set(split["strict_project_history_untouched_test_subset"]["case_ids"])
    strict_ok = {case_id for case_id in strict_ids if records[case_id]["status"] == "ok"}
    fact_counts = [len(record["facts"]) for record in records.values() if record["status"] == "ok"]
    weights = model_cache_dir / model_type / "weights.th"
    model_config = model_cache_dir / model_type / "config.json"
    if not weights.exists() or not model_config.exists():
        raise FileNotFoundError("RadGraph model weights/config missing from the frozen cache.")
    audit = {
        "audit": "V9 modern RadGraph XL preprocessing completion",
        "status": (
            "complete_no_errors"
            if status_counts["error"] == 0 and len(records) == config["source"]["case_count"]
            else "incomplete_or_errors"
        ),
        "config_path": portable_path(config_path),
        "config_sha256": file_sha256(config_path),
        "source_path": portable_path(cases_path),
        "source_sha256": file_sha256(cases_path),
        "split_path": portable_path(split_path),
        "split_sha256": file_sha256(split_path),
        "model": {
            "package": "radgraph",
            "package_version": package_version("radgraph"),
            "model_type": model_type,
            "weights_sha256": file_sha256(weights),
            "config_sha256": file_sha256(model_config),
            "foundation_parameters_updated": False,
        },
        "records": {
            "total": len(records),
            "ok": status_counts["ok"],
            "empty_report": status_counts["empty_report"],
            "error": status_counts["error"],
            "case_ids_sha256": case_id_fingerprint(records),
            "facts_sha256": canonical_fact_fingerprint(records.values()),
        },
        "by_partition": split_status,
        "primary_qrel_frames": {
            "shared_candidate_bank": split_status["train"]["ok"],
            "train_queries": split_status["train"]["ok"],
            "validation_queries": split_status["validation"]["ok"],
            "test_queries": split_status["test"]["ok"],
            "strict_untouched_test_queries": len(strict_ok),
            "strict_untouched_test_case_ids_sha256": case_id_fingerprint(strict_ok),
        },
        "fact_count_distribution_ok_records": {
            "minimum": min(fact_counts) if fact_counts else None,
            "median": statistics.median(fact_counts) if fact_counts else None,
            "mean": statistics.fmean(fact_counts) if fact_counts else None,
            "maximum": max(fact_counts) if fact_counts else None,
        },
        "local_output": {
            "path": portable_path(output_path),
            "sha256": file_sha256(output_path),
            "contains_report_derived_text": True,
            "committed_to_public_repository": False,
        },
        "elapsed_seconds_this_invocation": elapsed_seconds,
        "target_report_facts_available_to_inference": False,
        "v9_outcomes_inspected": False,
    }
    expected = config["primary_frames"]
    observed = audit["primary_qrel_frames"]
    checks = {
        "shared_candidate_bank": expected["shared_candidate_bank"],
        "train_queries": expected["train_qrel_queries"],
        "validation_queries": expected["validation_qrel_queries"],
        "test_queries": expected["test_qrel_queries"],
        "strict_untouched_test_queries": expected["strict_untouched_test_queries"],
    }
    if any(observed[name] != value for name, value in checks.items()):
        raise RuntimeError(f"Primary qrel frame counts changed: {observed!r}.")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate checkpointed V9 modern-RadGraph-XL annotations."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--model-type", default="modern-radgraph-xl")
    parser.add_argument("--model-cache-dir", type=Path, default=default_model_cache())
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--checkpoint-batch-size", type=int, default=8)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    config = read_json(args.config)
    if file_sha256(args.cases) != config["source"]["sha256"]:
        raise RuntimeError("Source SHA-256 differs from the frozen RadGraph protocol.")
    if file_sha256(args.split) != config["split"]["sha256"]:
        raise RuntimeError("Split SHA-256 differs from the frozen RadGraph protocol.")
    if args.model_type != config["model"]["model_type"]:
        raise RuntimeError("Requested RadGraph model differs from the frozen protocol.")
    if package_version("radgraph") != config["model"]["package_version"]:
        raise RuntimeError("Installed RadGraph package version differs from the protocol.")
    if args.checkpoint_batch_size <= 0:
        raise ValueError("checkpoint-batch-size must be positive.")

    rows = read_jsonl(args.cases)
    rows_by_id = {canonical_case_id(row["case_id"]): row for row in rows}
    if len(rows_by_id) != len(rows) or len(rows) != config["source"]["case_count"]:
        raise RuntimeError("Source case IDs are duplicated or source count changed.")
    texts = {case_id: normalized_report_text(row) for case_id, row in rows_by_id.items()}

    records = load_checkpoint(args.output)
    unknown = set(records) - set(rows_by_id)
    if unknown:
        raise RuntimeError(f"Checkpoint contains unknown source IDs: {sorted(unknown)[:5]}")
    for case_id, record in records.items():
        if record.get("model_type") != args.model_type:
            raise RuntimeError(f"Checkpoint model mismatch for {case_id}.")
        if record.get("report_text_sha256") != text_sha256(texts[case_id]):
            raise RuntimeError(f"Checkpoint source-text mismatch for {case_id}.")
    if args.retry_errors:
        records = {case_id: row for case_id, row in records.items() if row["status"] != "error"}
        canonicalize_output(args.output, records)

    pending_empty = [
        case_id for case_id in sorted(rows_by_id) if case_id not in records and not texts[case_id]
    ]
    if pending_empty:
        empty_records = [
            annotation_record(
                case_id=case_id,
                report_text="",
                model_type=args.model_type,
                annotation=None,
            )
            for case_id in pending_empty
        ]
        append_records(args.output, empty_records)
        records.update({row["case_id"]: row for row in empty_records})

    pending = [
        case_id for case_id in sorted(rows_by_id) if case_id not in records and texts[case_id]
    ]
    if pending:
        from radgraph import RadGraph

        model = RadGraph(
            model_type=args.model_type,
            cuda=args.cuda,
            batch_size=1,
            model_cache_dir=str(args.model_cache_dir),
        )
        total = len(pending)
        for offset in range(0, total, args.checkpoint_batch_size):
            batch_ids = pending[offset : offset + args.checkpoint_batch_size]
            batch_texts = [texts[case_id] for case_id in batch_ids]
            try:
                annotations = model(batch_texts)
                if len(annotations) != len(batch_ids):
                    raise RuntimeError(
                        f"RadGraph returned {len(annotations)} rows for {len(batch_ids)} inputs."
                    )
                batch_records = [
                    annotation_record(
                        case_id=case_id,
                        report_text=texts[case_id],
                        model_type=args.model_type,
                        annotation=annotations[str(index)],
                    )
                    for index, case_id in enumerate(batch_ids)
                ]
            except Exception as exc:
                batch_records = [
                    annotation_record(
                        case_id=case_id,
                        report_text=texts[case_id],
                        model_type=args.model_type,
                        annotation=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    for case_id in batch_ids
                ]
            append_records(args.output, batch_records)
            records.update({row["case_id"]: row for row in batch_records})
            completed = min(offset + len(batch_ids), total)
            elapsed = time.perf_counter() - started
            print(
                f"annotated={completed}/{total} checkpoint_records={len(records)}/{len(rows_by_id)} "
                f"elapsed_seconds={elapsed:.1f}",
                flush=True,
            )

    canonicalize_output(args.output, records)
    elapsed = time.perf_counter() - started
    audit = build_audit(
        config_path=args.config,
        cases_path=args.cases,
        split_path=args.split,
        output_path=args.output,
        records=records,
        model_type=args.model_type,
        model_cache_dir=args.model_cache_dir,
        elapsed_seconds=elapsed,
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=True))
    if audit["status"] != "complete_no_errors":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
