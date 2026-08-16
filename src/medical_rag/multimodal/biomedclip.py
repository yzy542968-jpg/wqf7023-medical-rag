from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


class BiomedClipEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        context_length: int = 256,
    ) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional model runtime
            raise RuntimeError(
                "BiomedCLIP requires the optional open_clip_torch and torch dependencies."
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        self.torch = torch
        self.device = torch.device(device)
        self.context_length = context_length
        self.model_name = model_name
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _normalized_numpy(tensor: Any) -> np.ndarray:
        tensor = tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return tensor.detach().float().cpu().numpy()

    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            tokens = self.tokenizer(batch, context_length=self.context_length).to(self.device)
            with self.torch.inference_mode():
                features = self.model.encode_text(tokens)
            outputs.append(self._normalized_numpy(features))
        if not outputs:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def encode_images(self, paths: Sequence[Path], batch_size: int = 16) -> np.ndarray:
        from PIL import Image

        outputs = []
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as image:
                    tensors.append(self.preprocess(image.convert("RGB")))
            batch = self.torch.stack(tensors).to(self.device)
            with self.torch.inference_mode():
                features = self.model.encode_image(batch)
            outputs.append(self._normalized_numpy(features))
        if not outputs:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(outputs, axis=0)
