from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _first(row: dict[str, str], *keys: str) -> str:
    normalized = {key.lower().strip(): value for key, value in row.items()}
    for key in keys:
        value = normalized.get(key.lower())
        if value:
            return value.strip()
    return ""


def _case_id(row: dict[str, str]) -> str:
    raw_id = _first(row, "uid", "id", "case_id")
    if raw_id.upper().startswith("CXR"):
        return raw_id.upper()
    return f"CXR{raw_id}" if raw_id else ""


def build_openi_cases(reports_csv: Path, projections_csv: Path | None = None) -> list[dict[str, Any]]:
    reports = _read_csv(reports_csv)
    image_map: dict[str, list[dict[str, str]]] = defaultdict(list)

    if projections_csv and projections_csv.exists():
        for row in _read_csv(projections_csv):
            case_id = _case_id(row)
            filename = _first(row, "filename", "image", "image_filename")
            projection = _first(row, "projection", "view")
            if case_id and filename:
                image_map[case_id].append(
                    {
                        "filename": filename,
                        "projection": projection,
                    }
                )

    cases: list[dict[str, Any]] = []
    for row in reports:
        case_id = _case_id(row)
        if not case_id:
            continue

        indication = _first(row, "indication")
        comparison = _first(row, "comparison")
        findings = _first(row, "findings")
        impression = _first(row, "impression")
        mesh = _first(row, "mesh", "MeSH")
        problems = _first(row, "problems")
        report_text = "\n".join(part for part in [indication, findings, impression] if part)

        cases.append(
            {
                "case_id": case_id,
                "indication": indication,
                "comparison": comparison,
                "findings": findings,
                "impression": impression,
                "mesh": mesh,
                "problems": problems,
                "report_text": report_text,
                "images": image_map.get(case_id, []),
            }
        )

    return cases


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized OpenI case JSONL.")
    parser.add_argument("--reports-csv", required=True, type=Path)
    parser.add_argument("--projections-csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cases = build_openi_cases(args.reports_csv, args.projections_csv)
    write_jsonl(cases, args.output)
    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()

