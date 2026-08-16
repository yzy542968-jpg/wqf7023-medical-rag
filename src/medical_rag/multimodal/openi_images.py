from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def official_filename_candidates(case_id: str, declared_filename: str) -> list[str]:
    """Map mirror projection names to the official NLM PNG naming scheme."""
    normalized = declared_filename
    if normalized.lower().endswith(".dcm.png"):
        normalized = normalized[: -len(".dcm.png")] + ".png"
    candidates = [f"CXR{normalized}", f"{case_id}_{normalized}", declared_filename]
    return list(dict.fromkeys(candidates))


def resolve_official_image(
    case_id: str,
    declared_filename: str,
    images_by_name: Mapping[str, Path],
) -> Path | None:
    for candidate in official_filename_candidates(case_id, declared_filename):
        if candidate in images_by_name:
            return images_by_name[candidate]
    return None
