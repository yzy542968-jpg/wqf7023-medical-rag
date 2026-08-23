from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def canonical_identifier(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized


@dataclass(frozen=True)
class PairedCase:
    """A source-neutral image-report study used by the V9 task contract."""

    study_id: str
    patient_id: str | None
    image_paths: tuple[str, ...]
    indication: str
    findings: str
    impression: str
    labels: Mapping[str, Any] = field(default_factory=dict)
    radgraph_facts: frozenset[str] = field(default_factory=frozenset)
    source: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "study_id",
            canonical_identifier(self.study_id, field_name="study_id"),
        )
        if self.patient_id is not None:
            object.__setattr__(
                self,
                "patient_id",
                canonical_identifier(self.patient_id, field_name="patient_id"),
            )
        normalized_paths = tuple(str(path).strip() for path in self.image_paths if str(path).strip())
        if not normalized_paths:
            raise ValueError("At least one image path is required.")
        object.__setattr__(self, "image_paths", normalized_paths)
        object.__setattr__(self, "indication", " ".join(str(self.indication).split()))
        object.__setattr__(self, "findings", " ".join(str(self.findings).split()))
        object.__setattr__(self, "impression", " ".join(str(self.impression).split()))
        object.__setattr__(
            self,
            "radgraph_facts",
            frozenset(" ".join(str(fact).lower().split()) for fact in self.radgraph_facts if str(fact).strip()),
        )

    @property
    def report_text(self) -> str:
        return "\n".join(part for part in (self.findings, self.impression) if part)

    def query_text(self, question: str) -> str:
        return "\n".join(
            part for part in (self.indication, " ".join(question.split())) if part
        )
