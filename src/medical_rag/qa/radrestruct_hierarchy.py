from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


YES = np.asarray([1, 0], dtype=np.int8)
NO = np.asarray([0, 1], dtype=np.int8)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class RadReStructHierarchy:
    """Apply the official Rad-ReStruct report-consistency rules.

    The traversal follows the MIT-licensed Rad-ReStruct evaluator while using
    explicit dataset paths and immutable caller inputs. This project computes
    its own metrics after cleaning instead of repeating the official evaluator's
    aggregate-row averaging behavior.
    """

    def __init__(self, dataset_root: str | Path):
        self.root = Path(dataset_root)
        self.report_keys = tuple(_load_json(self.root / "report_keys.json"))
        self.question_ids = np.asarray(
            _load_json(self.root / "vectorized_question_ids.json"), dtype=int
        )
        raw_choice_options = _load_json(
            self.root / "vectorized_choice_options.json"
        )
        self.choice_options = {
            int(question_id): str(choice)
            for question_id, choice in raw_choice_options.items()
        }
        self.max_instances = _load_json(self.root / "max_instances.json")
        self.template = _load_json(self.root / "template_final_clean.json")
        if len(self.report_keys) != len(self.question_ids):
            raise ValueError("Report keys and vectorized question IDs must align")
        self.report_labels = np.asarray(
            [self._answer_label(key) for key in self.report_keys], dtype=object
        )
        self.indices_by_question = {
            question_id: np.flatnonzero(self.question_ids == question_id)
            for question_id in sorted(set(self.question_ids.tolist()))
        }

    @staticmethod
    def _answer_label(key: str) -> str:
        pieces = key.split("_")
        if pieces[-1] in {"yes", "no"}:
            return pieces[-1]
        return pieces[-2]

    def _indices(self, question_id: int) -> np.ndarray:
        try:
            return self.indices_by_question[question_id]
        except KeyError as error:
            raise ValueError(f"Missing vectorized question ID: {question_id}") from error

    def _set_negative_or_empty(self, prediction: np.ndarray, question_id: int) -> None:
        indices = self._indices(question_id)
        labels = self.report_labels[indices]
        if "yes" in labels:
            prediction[indices] = NO
        else:
            prediction[indices] = 0

    def _iterate_area(
        self,
        area: Mapping[str, Any],
        area_name: str,
        prediction: np.ndarray,
        current_question_id: int,
    ) -> int:
        for topic_name, topic in area.items():
            if topic_name == "area":
                continue
            if topic_name == "infos":
                maximum = int(self.max_instances.get(f"{area_name}/{topic_name}", 1))
                previous = None
                for instance_index in range(maximum):
                    indices = self._indices(current_question_id)
                    if previous is not None and np.array_equal(previous, NO):
                        if instance_index == 0:
                            raise AssertionError("First info instance cannot follow a prior no")
                        prediction[indices] = NO
                    topic_prediction = prediction[indices].copy()
                    previous = topic_prediction
                    current_question_id += 1
                    for info_name in topic:
                        if info_name == "instances":
                            continue
                        if len(topic_prediction) >= 2 and topic_prediction[1] == 1:
                            self._set_negative_or_empty(
                                prediction, current_question_id
                            )
                        current_question_id += 1
                continue

            area_indices = self._indices(current_question_id)
            area_prediction = prediction[area_indices].copy()
            current_question_id += 1
            for element_name, element in topic.items():
                maximum = int(
                    self.max_instances.get(f"{area_name}/{element_name}", 1)
                )
                previous = None
                for _ in range(maximum):
                    indices = self._indices(current_question_id)
                    if (
                        len(area_prediction) >= 2
                        and area_prediction[1] == 1
                    ) or (previous is not None and np.array_equal(previous, NO)):
                        prediction[indices] = NO
                    topic_prediction = prediction[indices].copy()
                    previous = topic_prediction
                    current_question_id += 1
                    for _info_name in element["infos"]:
                        area_yes = len(area_prediction) >= 1 and area_prediction[0] == 1
                        topic_yes = (
                            len(topic_prediction) >= 1 and topic_prediction[0] == 1
                        )
                        if not (area_yes and topic_yes):
                            self._set_negative_or_empty(
                                prediction, current_question_id
                            )
                        current_question_id += 1
        return current_question_id

    def _adapt_row(self, prediction: np.ndarray) -> np.ndarray:
        current_question_id = 0
        for area in self.template:
            if "sub_areas" in area:
                for sub_area_name, sub_area in area["sub_areas"].items():
                    current_question_id = self._iterate_area(
                        sub_area,
                        sub_area_name,
                        prediction,
                        current_question_id,
                    )
            else:
                current_question_id = self._iterate_area(
                    area,
                    str(area["area"]),
                    prediction,
                    current_question_id,
                )
        return prediction

    def clean(self, predictions: np.ndarray) -> np.ndarray:
        matrix = np.asarray(predictions)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.report_keys):
            raise ValueError(
                f"Predictions must have shape (cases, {len(self.report_keys)})"
            )
        if not np.isin(matrix, (0, 1)).all():
            raise ValueError("Predictions must be binary before hierarchy cleaning")
        cleaned = matrix.astype(np.uint8, copy=True)
        for row in cleaned:
            for question_id, choice in self.choice_options.items():
                indices = self._indices(question_id)
                if choice == "single_choice":
                    if int(row[indices].sum()) > 1:
                        raise ValueError(
                            f"Single-choice question {question_id} has multiple answers"
                        )
                elif choice == "multi_choice":
                    labels = self.report_labels[indices]
                    fallback_index = None
                    for fallback_label in ("unspecified", "no selection"):
                        matches = np.flatnonzero(labels == fallback_label)
                        if len(matches):
                            fallback_index = int(indices[int(matches[0])])
                            break
                    selected = int(row[indices].sum())
                    if selected > 1 and fallback_index is not None:
                        row[fallback_index] = 0
                    elif selected == 0 and fallback_index is not None:
                        row[fallback_index] = 1
            self._adapt_row(row)
        return cleaned
