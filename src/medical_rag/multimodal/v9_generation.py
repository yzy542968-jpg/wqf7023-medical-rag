from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from medical_rag.multimodal.openi_images import official_filename_candidates
from medical_rag.multimodal.v6_generation import (
    MEDGEMMA_MODEL,
    MEDGEMMA_REVISION,
    MedGemmaTextGenerator,
    normalized_text,
)


def select_primary_image(case: Mapping[str, Any], image_root: Path) -> Path:
    images = list(case.get("images") or [])
    if not images:
        raise ValueError(f"Case {case.get('case_id')} has no image metadata.")
    ordered = sorted(
        images,
        key=lambda item: (
            0 if "frontal" in str(item.get("projection", "")).lower() else 1,
            str(item.get("filename", "")),
        ),
    )
    case_id = str(case.get("case_id", ""))
    filename = str(ordered[0].get("filename", ""))
    for candidate in official_filename_candidates(case_id, filename):
        path = image_root / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not resolve the primary image for {case_id}: {filename}")


def build_v9_qa_prompt(
    question: str,
    indication: str,
    retrieved_cases: Sequence[Mapping[str, Any]],
) -> str:
    evidence = []
    for index, case in enumerate(retrieved_cases, start=1):
        evidence.extend(
            [
                f"Historical case {index} ID: {normalized_text(case.get('case_id'))}",
                f"Findings: {normalized_text(case.get('findings')) or 'Not documented'}",
                f"Impression: {normalized_text(case.get('impression')) or 'Not documented'}",
            ]
        )
    if not evidence:
        evidence = ["No historical cases were provided."]
    return "\n".join(
        [
            "You are a careful radiology question-answering assistant in a research experiment.",
            "Use the target chest radiograph as the primary patient evidence.",
            "Historical reports are other-patient analogies, not proof that a finding is present in the target patient.",
            "Do not claim that a historical finding belongs to the target unless it is also visible in the target image.",
            "If the target image is insufficient, state uncertainty or abstain.",
            "Return exactly one JSON object and no markdown.",
            "Keep answer to at most two concise sentences, target_image_findings to at most three short items, and historical_support to one sentence.",
            'Schema: {"answer":"...","target_image_findings":["..."],"supporting_case_ids":["..."],"historical_support":"...","uncertainty":"low|medium|high","abstain":false}',
            "Use only IDs listed below in supporting_case_ids. Use an empty list when no historical case supports the answer.",
            "",
            f"Clinical indication: {normalized_text(indication) or 'Not provided'}",
            f"Question: {normalized_text(question)}",
            "",
            "Retrieved other-patient evidence:",
            *evidence,
        ]
    )


def parse_v9_output(text: str, allowed_case_ids: Sequence[str]) -> dict[str, Any]:
    raw = str(text or "").strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    parsed: Any = None
    if match:
        try:
            parsed = json.loads(match.group(0), strict=False)
        except json.JSONDecodeError:
            parsed = None
    valid = isinstance(parsed, dict)
    if not valid:
        parsed = {}
    allowed = set(map(str, allowed_case_ids))
    ids = parsed.get("supporting_case_ids", [])
    ids = ids if isinstance(ids, list) else []
    ids = [str(value) for value in ids if str(value) in allowed]
    findings = parsed.get("target_image_findings", [])
    findings = findings if isinstance(findings, list) else []
    uncertainty = str(parsed.get("uncertainty", "high")).lower()
    if uncertainty not in {"low", "medium", "high"}:
        uncertainty = "high"
        valid = False
    answer = normalized_text(parsed.get("answer", ""))
    if not answer:
        answer = normalized_text(raw)
        valid = False
    return {
        "answer": answer,
        "target_image_findings": [normalized_text(value) for value in findings if normalized_text(value)],
        "supporting_case_ids": ids,
        "historical_support": normalized_text(parsed.get("historical_support", "")),
        "uncertainty": uncertainty,
        "abstain": bool(parsed.get("abstain", False)),
        "structured_output_valid": valid,
        "raw_output": raw,
    }


class MedGemmaImageGenerator(MedGemmaTextGenerator):
    def generate_batch(
        self,
        prompts: Sequence[str],
        image_paths: Sequence[Path],
        *,
        max_new_tokens: int = 192,
    ) -> list[dict[str, Any]]:
        if len(prompts) != len(image_paths):
            raise ValueError("prompts and image_paths must have equal length.")
        messages = []
        images = []
        for prompt, path in zip(prompts, image_paths, strict=True):
            image = Image.open(path).convert("RGB")
            images.append(image)
            messages.append(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            )
        rendered = [
            self.processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
            for message in messages
        ]
        inputs = self.processor(
            text=rendered,
            images=[[image] for image in images],
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        prompt_width = int(inputs["input_ids"].shape[1])
        results = []
        for row, input_length in zip(generated, input_lengths, strict=True):
            answer_ids = row[prompt_width:]
            results.append(
                {
                    "answer": self.processor.decode(answer_ids, skip_special_tokens=True).strip(),
                    "input_tokens": int(input_length),
                    "output_tokens": int(answer_ids.shape[-1]),
                }
            )
        for image in images:
            image.close()
        return results


__all__ = [
    "MEDGEMMA_MODEL",
    "MEDGEMMA_REVISION",
    "MedGemmaImageGenerator",
    "build_v9_qa_prompt",
    "parse_v9_output",
    "select_primary_image",
]
