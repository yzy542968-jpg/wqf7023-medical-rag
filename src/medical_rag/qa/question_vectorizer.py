from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from medical_rag.qa.radrestruct import RadReStructQuestion
from medical_rag.qa.radrestruct_hierarchy import RadReStructHierarchy


class RadReStructQuestionVectorizer:
    """Map ordered independent QA rows into the official report-vector space."""

    def __init__(self, hierarchy: RadReStructHierarchy):
        self.hierarchy = hierarchy
        self.options_by_question = {
            question_id: tuple(
                str(hierarchy.report_labels[index])
                for index in hierarchy.indices_by_question[question_id]
            )
            for question_id in hierarchy.indices_by_question
        }
        self.keys_by_question = {
            question_id: tuple(
                hierarchy.report_keys[index]
                for index in hierarchy.indices_by_question[question_id]
            )
            for question_id in hierarchy.indices_by_question
        }

    def question_ids(
        self, questions: Sequence[RadReStructQuestion]
    ) -> tuple[int, ...]:
        mapped: list[int] = []
        previous = -1
        for row in questions:
            candidates = [
                question_id
                for question_id in range(previous + 1, len(self.options_by_question))
                if self.options_by_question[question_id] == row.options
                and all(
                    self._path_matches(key, row.path)
                    for key in self.keys_by_question[question_id]
                )
            ]
            if not candidates:
                raise ValueError(
                    "Could not align ordered QA row to report space: "
                    f"path={row.path!r}, options={row.options!r}, previous_qid={previous}"
                )
            previous = candidates[0]
            mapped.append(previous)
        return tuple(mapped)

    @staticmethod
    def _path_matches(report_key: str, path: str) -> bool:
        """Allow instance names inserted between stable hierarchy path parts."""

        key_parts = report_key.split("_")
        position = 0
        for path_part in path.split("_"):
            try:
                position = key_parts.index(path_part, position) + 1
            except ValueError:
                return False
        return True

    def vectorize_answers(
        self,
        questions: Sequence[RadReStructQuestion],
        answers: Sequence[Sequence[str]] | None = None,
    ) -> np.ndarray:
        if answers is None:
            answers = [row.answers for row in questions]
        if len(answers) != len(questions):
            raise ValueError("One answer sequence is required for every question")
        vector = np.zeros((1, len(self.hierarchy.report_keys)), dtype=np.uint8)
        for question_id, row_answers in zip(
            self.question_ids(questions), answers, strict=True
        ):
            indices = self.hierarchy.indices_by_question[question_id]
            labels = self.options_by_question[question_id]
            requested = {" ".join(str(answer).split()) for answer in row_answers}
            unknown = requested - set(labels)
            if unknown:
                raise ValueError(
                    f"Answers are not valid for question {question_id}: {sorted(unknown)}"
                )
            for index, label in zip(indices, labels, strict=True):
                vector[0, index] = label in requested
        return self.hierarchy.clean(vector)[0]


__all__ = ["RadReStructQuestionVectorizer"]
