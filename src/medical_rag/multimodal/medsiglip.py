from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL = "google/medsiglip-448"
DEFAULT_REVISION = "9cea28a1a1195f665105faa6e8544c112fd960a4"


class MedSiglipEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        revision: str = DEFAULT_REVISION,
        device: str = "cuda",
        cache_dir: Path | None = None,
        max_text_tokens: int = 64,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional model runtime
            raise RuntimeError(
                "MedSigLIP requires torch, transformers, Pillow, and sentencepiece."
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        self.torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.max_text_tokens = int(max_text_tokens)
        kwargs: dict[str, Any] = {
            "revision": revision,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "local_files_only": local_files_only,
        }
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            use_fast=False,
            **kwargs,
        )
        configured_max = int(self.processor.tokenizer.model_max_length)
        if configured_max != self.max_text_tokens:
            raise RuntimeError(
                f"MedSigLIP tokenizer limit changed: expected {self.max_text_tokens}, "
                f"found {configured_max}."
            )
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            **kwargs,
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _normalized_numpy(tensor: Any) -> np.ndarray:
        tensor = tensor.float()
        tensor = tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return tensor.detach().cpu().numpy()

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            for text in batch_texts:
                tokens = self.processor.tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                    verbose=False,
                )["input_ids"]
                if len(tokens) > self.max_text_tokens:
                    raise ValueError(
                        f"MedSigLIP text contains {len(tokens)} tokens; "
                        f"maximum is {self.max_text_tokens}."
                    )
            encoded = self.processor(
                text=batch_texts,
                padding="max_length",
                truncation=False,
                max_length=self.max_text_tokens,
                return_tensors="pt",
            )
            model_inputs = {
                key: value.to(self.device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "position_ids"}
            }
            with self.torch.inference_mode():
                features = self.model.get_text_features(**model_inputs)
            outputs.append(self._normalized_numpy(features))
        if not outputs:
            return np.empty((0, 1152), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def encode_images(
        self,
        paths: Sequence[Path],
        *,
        batch_size: int = 2,
    ) -> np.ndarray:
        from PIL import Image

        outputs = []
        for start in range(0, len(paths), batch_size):
            images = []
            try:
                for path in paths[start : start + batch_size]:
                    images.append(Image.open(path).convert("RGB"))
                encoded = self.processor(images=images, return_tensors="pt")
                pixel_values = encoded["pixel_values"].to(
                    self.device, dtype=self.dtype
                )
                with self.torch.inference_mode():
                    features = self.model.get_image_features(
                        pixel_values=pixel_values
                    )
                outputs.append(self._normalized_numpy(features))
            finally:
                for image in images:
                    image.close()
        if not outputs:
            return np.empty((0, 1152), dtype=np.float32)
        return np.concatenate(outputs, axis=0)
