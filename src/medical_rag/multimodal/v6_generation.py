from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


MEDGEMMA_MODEL = "google/medgemma-1.5-4b-it"
MEDGEMMA_REVISION = "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def build_v6_qa_prompt(
    question: Mapping[str, Any],
    source_case: Mapping[str, Any],
    retrieved_case: Mapping[str, Any] | None,
) -> str:
    indication = normalized_text(source_case.get("indication")) or "Not provided"
    if retrieved_case is None:
        findings = "No report was retrieved."
        impression = "No report was retrieved."
    else:
        findings = normalized_text(retrieved_case.get("findings")) or "Not documented"
        impression = normalized_text(retrieved_case.get("impression")) or "Not documented"
    return "\n".join(
        [
            "You are a careful medical question-answering assistant in a research experiment.",
            "Answer using only the selected radiology report evidence below.",
            "Do not add unsupported findings, diagnoses, locations, severity, or certainty.",
            "If the selected report does not contain enough evidence, answer: Insufficient evidence.",
            "Return only the concise answer, without analysis or a preamble.",
            "",
            f"Clinical indication: {indication}",
            f"Question: {normalized_text(question.get('question'))}",
            "",
            "Selected report evidence:",
            f"Findings: {findings}",
            f"Impression: {impression}",
        ]
    )


class MedGemmaTextGenerator:
    def __init__(
        self,
        *,
        model_name: str = MEDGEMMA_MODEL,
        revision: str = MEDGEMMA_REVISION,
        cache_dir: Path | None = None,
        local_files_only: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ImportError as exc:  # pragma: no cover - optional model runtime
            raise RuntimeError(
                "MedGemma requires torch, transformers, accelerate, and bitsandbytes."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("The frozen MedGemma NF4 policy requires CUDA.")

        self.torch = torch
        kwargs = {
            "revision": revision,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "local_files_only": local_files_only,
        }
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False, **kwargs)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            quantization_config=quantization,
            device_map="auto",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            **kwargs,
        )
        self.model.eval()

    def generate_one(self, prompt: str, *, max_new_tokens: int = 256) -> dict[str, Any]:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = self.model.device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_tokens = int(inputs["input_ids"].shape[-1])
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
        answer_ids = generated[0, input_tokens:]
        answer = self.processor.decode(answer_ids, skip_special_tokens=True).strip()
        return {
            "answer": answer,
            "input_tokens": input_tokens,
            "output_tokens": int(answer_ids.shape[-1]),
        }


class QwenTextGenerator:
    def __init__(
        self,
        *,
        model_name: str = QWEN_MODEL,
        revision: str = QWEN_REVISION,
        cache_dir: Path | None = None,
        local_files_only: bool = True,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional model runtime
            raise RuntimeError("Qwen generation requires torch and transformers.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("The frozen Qwen FP16 policy requires CUDA.")

        self.torch = torch
        kwargs = {
            "revision": revision,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "local_files_only": local_files_only,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            **kwargs,
        ).to("cuda")
        self.model.eval()

    def generate_one(self, prompt: str, *, max_new_tokens: int = 256) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        answer_ids = generated[0, input_tokens:]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        return {
            "answer": answer,
            "input_tokens": input_tokens,
            "output_tokens": int(answer_ids.shape[-1]),
        }


def select_preflight_qids(qids: Sequence[str], count: int = 3) -> list[str]:
    if count <= 0:
        raise ValueError("count must be positive.")
    unique = sorted(set(qids))
    if len(unique) < count:
        raise ValueError("Not enough unique qids for the requested preflight sample.")
    if count == 1:
        return [unique[len(unique) // 2]]
    indices = [round(index * (len(unique) - 1) / (count - 1)) for index in range(count)]
    return [unique[index] for index in indices]
