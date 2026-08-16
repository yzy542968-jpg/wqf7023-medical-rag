from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL = "microsoft/BiomedVLP-BioViL-T"
DEFAULT_TEXT_REVISION = "692f09e"
IMAGE_WEIGHTS_FILENAME = "biovil_t_image_model_proj_size_128.pt"
IMAGE_WEIGHTS_MD5 = "a83080e2f23aa584a4f2b24c39b1bb64"


class BioVilTEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        text_revision: str = DEFAULT_TEXT_REVISION,
        device: str = "cuda",
        text_max_length: int = 256,
    ) -> None:
        try:
            import torch
            from health_multimodal.image.utils import ImageModelType, get_image_inference
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional model runtime
            raise RuntimeError(
                "BioViL-T requires torch, transformers, torchvision, and hi-ml-multimodal."
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        self.torch = torch
        self.device = torch.device(device)
        self.text_max_length = text_max_length

        self.image_engine = get_image_inference(ImageModelType.BIOVIL_T)
        self.image_engine.to(self.device)
        self._verify_image_weights()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=text_revision,
            trust_remote_code=True,
        )
        self.text_model = AutoModel.from_pretrained(
            model_name,
            revision=text_revision,
            trust_remote_code=True,
            use_safetensors=True,
        ).to(self.device)
        self.text_model.eval()

    @staticmethod
    def _verify_image_weights() -> None:
        path = Path(tempfile.gettempdir()) / IMAGE_WEIGHTS_FILENAME
        if not path.exists():
            raise RuntimeError(f"BioViL-T image weights were not downloaded to {path}.")
        actual = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - upstream checksum
        if actual != IMAGE_WEIGHTS_MD5:
            raise RuntimeError(f"BioViL-T image weight MD5 mismatch: {actual}")

    @staticmethod
    def _normalized_numpy(tensor: Any) -> np.ndarray:
        tensor = tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return tensor.detach().float().cpu().numpy()

    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        outputs = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=list(texts[start : start + batch_size]),
                add_special_tokens=True,
                padding="longest",
                truncation=True,
                max_length=self.text_max_length,
                return_tensors="pt",
            )
            with self.torch.inference_mode():
                features = self.text_model.get_projected_text_embeddings(
                    input_ids=tokens.input_ids.to(self.device),
                    attention_mask=tokens.attention_mask.to(self.device),
                )
            outputs.append(self._normalized_numpy(features))
        if not outputs:
            return np.empty((0, 128), dtype=np.float32)
        return np.concatenate(outputs, axis=0)

    def encode_images(self, paths: Sequence[Path], batch_size: int = 8) -> np.ndarray:
        from health_multimodal.image.data.io import load_image

        outputs = []
        for start in range(0, len(paths), batch_size):
            tensors = [
                self.image_engine.transform(load_image(path))
                for path in paths[start : start + batch_size]
            ]
            batch = self.torch.stack(tensors).to(self.device)
            with self.torch.inference_mode():
                features = self.image_engine.model(batch).projected_global_embedding
            outputs.append(self._normalized_numpy(features))
        if not outputs:
            return np.empty((0, 128), dtype=np.float32)
        return np.concatenate(outputs, axis=0)
