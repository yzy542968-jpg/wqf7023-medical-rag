from __future__ import annotations

import csv
import gzip
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, TextIO


SECTION_PATTERN = re.compile(
    r"(?ims)^\s*(findings|impression)\s*:\s*(.*?)(?=^\s*(?:findings|impression)\s*:|\Z)"
)


@dataclass(frozen=True)
class MimicCxrCase:
    subject_id: str
    study_id: str
    image_ids: tuple[str, ...]
    image_paths: tuple[str, ...]
    findings: str
    impression: str
    official_split: str | None


def _open_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_rows(path: Path) -> list[dict[str, str]]:
    with _open_csv(path) as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_report_sections(text: str) -> tuple[str, str]:
    sections = {
        match.group(1).lower(): " ".join(match.group(2).split())
        for match in SECTION_PATTERN.finditer(str(text or ""))
    }
    return sections.get("findings", ""), sections.get("impression", "")


def report_path(root: Path, subject_id: str, study_id: str) -> Path:
    subject = str(subject_id).removeprefix("p")
    study = str(study_id).removeprefix("s")
    return root / f"p{subject[:2]}" / f"p{subject}" / f"s{study}.txt"


def image_path(root: Path, subject_id: str, study_id: str, image_id: str) -> Path:
    subject = str(subject_id).removeprefix("p")
    study = str(study_id).removeprefix("s")
    return root / f"p{subject[:2]}" / f"p{subject}" / f"s{study}" / f"{image_id}.jpg"


def read_mimic_cxr_cases(
    *,
    record_list_csv: Path,
    report_root: Path,
    image_root: Path,
    split_csv: Path | None = None,
    require_images: bool = True,
) -> list[MimicCxrCase]:
    if not record_list_csv.exists():
        raise FileNotFoundError(record_list_csv)
    split_by_image: dict[str, str] = {}
    if split_csv is not None:
        for row in read_rows(split_csv):
            split_by_image[str(row.get("dicom_id", "")).strip()] = str(row.get("split", "")).strip()

    grouped: dict[tuple[str, str], list[str]] = {}
    for row in read_rows(record_list_csv):
        subject = str(row.get("subject_id", "")).strip().removeprefix("p")
        study = str(row.get("study_id", "")).strip().removeprefix("s")
        image = str(row.get("dicom_id", "")).strip()
        if not subject or not study or not image:
            raise ValueError("MIMIC record row lacks subject_id, study_id, or dicom_id")
        grouped.setdefault((subject, study), []).append(image)

    cases = []
    for (subject, study), image_ids in sorted(grouped.items()):
        report = report_path(report_root, subject, study)
        if not report.is_file():
            raise FileNotFoundError(report)
        findings, impression = parse_report_sections(report.read_text(encoding="utf-8"))
        paths = tuple(str(image_path(image_root, subject, study, image)) for image in sorted(set(image_ids)))
        if require_images:
            missing = [path for path in paths if not Path(path).is_file()]
            if missing:
                raise FileNotFoundError(missing[0])
        image_splits = {split_by_image.get(image, "") for image in image_ids} - {""}
        if len(image_splits) > 1:
            raise ValueError(f"Study {study} crosses official image splits")
        cases.append(
            MimicCxrCase(
                subject_id=subject,
                study_id=study,
                image_ids=tuple(sorted(set(image_ids))),
                image_paths=paths,
                findings=findings,
                impression=impression,
                official_split=next(iter(image_splits)) if image_splits else None,
            )
        )
    return cases


def patient_disjoint_partition(
    cases: Iterable[MimicCxrCase],
    fractions: Mapping[str, float],
    *,
    seed: int,
    domain: str = "v10-mimic-patient-split",
) -> dict[str, list[str]]:
    if abs(sum(float(value) for value in fractions.values()) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to one")
    subjects = sorted({case.subject_id for case in cases})
    ordered = sorted(
        subjects,
        key=lambda subject: (
            hashlib.sha256(f"{domain}|{seed}|{subject}".encode("utf-8")).hexdigest(),
            subject,
        ),
    )
    result = {name: [] for name in fractions}
    cumulative = 0.0
    boundaries = []
    for name, fraction in fractions.items():
        cumulative += float(fraction)
        boundaries.append((name, cumulative))
    for index, subject in enumerate(ordered):
        position = (index + 0.5) / max(len(ordered), 1)
        partition = next(name for name, boundary in boundaries if position <= boundary)
        result[partition].append(subject)
    return result


__all__ = [
    "MimicCxrCase",
    "image_path",
    "parse_report_sections",
    "patient_disjoint_partition",
    "read_mimic_cxr_cases",
    "report_path",
]

