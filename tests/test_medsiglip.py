from __future__ import annotations

import numpy as np
import pytest

from medical_rag.multimodal.medsiglip import MedSiglipEncoder


def test_medsiglip_normalization_returns_unit_float32_vectors() -> None:
    values = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float16)

    class TensorAdapter:
        def __init__(self, array: np.ndarray) -> None:
            import torch

            self.tensor = torch.as_tensor(array)

    tensor = TensorAdapter(values).tensor
    normalized = MedSiglipEncoder._normalized_numpy(tensor)
    assert normalized.dtype == np.float32
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)


def test_medsiglip_rejects_overlength_text_without_loading_model() -> None:
    encoder = object.__new__(MedSiglipEncoder)
    encoder.max_text_tokens = 2

    class Tokenizer:
        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, list[int]]:
            return {"input_ids": [1, 2, 3]}

    class Processor:
        tokenizer = Tokenizer()

    encoder.processor = Processor()

    with pytest.raises(ValueError, match="maximum is 2"):
        encoder.encode_texts(["too long"])
