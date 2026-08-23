from __future__ import annotations

import hashlib
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


PARTITION_ORDER = ("train", "calibration", "validation", "test")
SPECTRUM_ORDER = ("normal", "abnormal", "indeterminate")


def canonical_case_id(value: object) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError("case_id cannot be empty")
    return value


def normalized_report_text(findings: object, impression: object) -> str:
    joined = f"{findings or ''} {impression or ''}"
    return " ".join(unicodedata.normalize("NFKC", joined).lower().split())


def report_index_spectrum(problems: object) -> str:
    normalized = " ".join(str(problems or "").lower().split())
    if normalized == "normal":
        return "normal"
    if not normalized or normalized == "no indexing":
        return "indeterminate"
    return "abnormal"


def canonical_fingerprint(values: Iterable[str]) -> str:
    canonical = sorted({canonical_case_id(value) for value in values})
    return hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        canonical = [canonical_case_id(value) for value in values]
        if len(canonical) != len(set(canonical)):
            raise ValueError("UnionFind values must be unique")
        self.parent = {value: value for value in canonical}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        keep, merge = sorted((left_root, right_root))
        self.parent[merge] = keep

    def components(self) -> list[tuple[str, ...]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for value in sorted(self.parent):
            groups[self.find(value)].append(value)
        return sorted(
            (tuple(sorted(values)) for values in groups.values()),
            key=lambda values: values[0],
        )


@dataclass(frozen=True)
class DuplicateClusters:
    clusters: tuple[tuple[str, ...], ...]
    exact_text_edges: int
    near_text_edges: int
    exact_image_edges: int


def build_duplicate_clusters(
    case_ids: Sequence[str],
    report_texts: Sequence[str],
    *,
    cosine_threshold: float = 0.95,
    image_sha256_by_case: Mapping[str, Sequence[str]] | None = None,
) -> DuplicateClusters:
    if len(case_ids) != len(report_texts):
        raise ValueError("case_ids and report_texts must have equal length")
    if not 0.0 < cosine_threshold <= 1.0:
        raise ValueError("cosine_threshold must be in (0, 1]")
    canonical_ids = [canonical_case_id(value) for value in case_ids]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("case_ids must be unique")
    text_by_id = dict(zip(canonical_ids, report_texts, strict=True))
    union = UnionFind(canonical_ids)

    exact_text_edges = 0
    first_by_text: dict[str, str] = {}
    for case_id, text in text_by_id.items():
        if not text:
            continue
        first = first_by_text.setdefault(text, case_id)
        if first != case_id:
            union.union(first, case_id)
            exact_text_edges += 1

    nonempty = [(case_id, text) for case_id, text in text_by_id.items() if text]
    near_text_edges = 0
    if len(nonempty) >= 2:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            lowercase=False,
            dtype=np.float32,
            norm="l2",
        )
        matrix = vectorizer.fit_transform([text for _, text in nonempty])
        similarity = (matrix @ matrix.T).tocoo()
        for row, col, score in zip(similarity.row, similarity.col, similarity.data, strict=True):
            if row >= col or float(score) + 1e-7 < cosine_threshold:
                continue
            left = nonempty[int(row)][0]
            right = nonempty[int(col)][0]
            if text_by_id[left] == text_by_id[right]:
                continue
            union.union(left, right)
            near_text_edges += 1

    exact_image_edges = 0
    first_by_image: dict[str, str] = {}
    for case_id in canonical_ids:
        for digest in sorted(set((image_sha256_by_case or {}).get(case_id, ()))):
            first = first_by_image.setdefault(str(digest), case_id)
            if first != case_id:
                union.union(first, case_id)
                exact_image_edges += 1

    return DuplicateClusters(
        clusters=tuple(union.components()),
        exact_text_edges=exact_text_edges,
        near_text_edges=near_text_edges,
        exact_image_edges=exact_image_edges,
    )


def _hash_order(domain: str, seed: int, cluster_id: str) -> tuple[str, str]:
    payload = f"{domain}|{seed}|{cluster_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), cluster_id


def assign_clusters(
    clusters: Sequence[Sequence[str]],
    spectrum_by_case: Mapping[str, str],
    fractions: Mapping[str, float],
    *,
    domain: str,
    seed: int,
) -> dict[str, list[str]]:
    if tuple(fractions) != PARTITION_ORDER:
        raise ValueError(f"fractions must follow {PARTITION_ORDER}")
    if abs(sum(float(value) for value in fractions.values()) - 1.0) > 1e-9:
        raise ValueError("partition fractions must sum to one")

    flat = [canonical_case_id(case_id) for cluster in clusters for case_id in cluster]
    if len(flat) != len(set(flat)):
        raise ValueError("clusters must be case-disjoint")
    if set(flat) != set(spectrum_by_case):
        raise ValueError("clusters and spectrum map must cover the same cases")
    if any(value not in SPECTRUM_ORDER for value in spectrum_by_case.values()):
        raise ValueError("unknown spectrum label")

    total_by_spectrum = Counter(spectrum_by_case.values())
    targets: dict[str, dict[str, float]] = {}
    for partition in PARTITION_ORDER:
        targets[partition] = {
            "total": len(flat) * float(fractions[partition]),
            **{
                spectrum: total_by_spectrum[spectrum] * float(fractions[partition])
                for spectrum in SPECTRUM_ORDER
            },
        }

    counts = {
        partition: {"total": 0, **{spectrum: 0 for spectrum in SPECTRUM_ORDER}}
        for partition in PARTITION_ORDER
    }
    assignments = {partition: [] for partition in PARTITION_ORDER}
    ordered = sorted(
        (tuple(sorted(map(canonical_case_id, cluster))) for cluster in clusters),
        key=lambda cluster: _hash_order(domain, seed, cluster[0]),
    )

    for cluster in ordered:
        cluster_counts = Counter(spectrum_by_case[case_id] for case_id in cluster)
        cluster_counts["total"] = len(cluster)
        candidates: list[tuple[float, int, str]] = []
        for order_index, partition in enumerate(PARTITION_ORDER):
            penalty = 0.0
            for candidate_partition in PARTITION_ORDER:
                for key in ("total", *SPECTRUM_ORDER):
                    projected = counts[candidate_partition][key]
                    if candidate_partition == partition:
                        projected += cluster_counts[key]
                    target = targets[candidate_partition][key]
                    penalty += ((projected - target) ** 2) / max(target, 1.0)
            candidates.append((penalty, order_index, partition))
        partition = min(candidates)[2]
        assignments[partition].extend(cluster)
        for key in ("total", *SPECTRUM_ORDER):
            counts[partition][key] += int(cluster_counts[key])

    return {partition: sorted(values) for partition, values in assignments.items()}


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DuplicateClusters",
    "PARTITION_ORDER",
    "SPECTRUM_ORDER",
    "UnionFind",
    "assign_clusters",
    "build_duplicate_clusters",
    "canonical_case_id",
    "canonical_fingerprint",
    "file_sha256",
    "normalized_report_text",
    "report_index_spectrum",
]

