from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


QUERY_ENCODER_NAME = "ncbi/MedCPT-Query-Encoder"
ARTICLE_ENCODER_NAME = "ncbi/MedCPT-Article-Encoder"


def _require_transformer_stack():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "MedCPT retrieval requires optional dependencies: torch and transformers. "
            "Install the dense retrieval dependencies before running this script."
        ) from exc
    return torch, AutoModel, AutoTokenizer


def _batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _case_article_pair(case: dict[str, Any]) -> list[str]:
    title_parts = [
        case.get("indication", ""),
        case.get("problems", ""),
    ]
    abstract_parts = [
        case.get("findings", ""),
        case.get("impression", ""),
    ]
    title = " ".join(part for part in title_parts if part).strip() or case["case_id"]
    abstract = " ".join(part for part in abstract_parts if part).strip() or case.get("report_text", "")
    return [title, abstract]


def encode_queries(
    queries: list[str],
    model_name: str = QUERY_ENCODER_NAME,
    batch_size: int = 16,
    device: str | None = None,
    max_length: int = 64,
    local_files_only: bool = False,
) -> np.ndarray:
    torch, AutoModel, AutoTokenizer = _require_transformer_stack()
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(selected_device)
    model.eval()

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _batched(queries, batch_size):
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=max_length,
            )
            encoded = {key: value.to(selected_device) for key, value in encoded.items()}
            embeddings = model(**encoded).last_hidden_state[:, 0, :]
            vectors.append(embeddings.detach().cpu().numpy().astype("float32"))

    return _l2_normalize(np.vstack(vectors))


def encode_cases(
    cases: list[dict[str, Any]],
    model_name: str = ARTICLE_ENCODER_NAME,
    batch_size: int = 8,
    device: str | None = None,
    max_length: int = 512,
) -> np.ndarray:
    torch, AutoModel, AutoTokenizer = _require_transformer_stack()
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(selected_device)
    model.eval()

    article_pairs = [_case_article_pair(case) for case in cases]
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _batched(article_pairs, batch_size):
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=max_length,
            )
            encoded = {key: value.to(selected_device) for key, value in encoded.items()}
            embeddings = model(**encoded).last_hidden_state[:, 0, :]
            vectors.append(embeddings.detach().cpu().numpy().astype("float32"))

    return _l2_normalize(np.vstack(vectors))


def build_medcpt_index(
    cases_path: Path,
    output_path: Path,
    batch_size: int = 8,
    device: str | None = None,
    limit: int | None = None,
) -> None:
    cases = load_cases_jsonl(cases_path)
    if limit is not None:
        cases = cases[:limit]
    embeddings = encode_cases(cases, batch_size=batch_size, device=device)
    case_ids = np.array([case["case_id"] for case in cases])
    metadata = {
        "cases_path": str(cases_path),
        "case_count": len(cases),
        "article_encoder": ARTICLE_ENCODER_NAME,
        "query_encoder": QUERY_ENCODER_NAME,
        "embedding_dim": int(embeddings.shape[1]),
        "normalized": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        case_ids=case_ids,
        metadata=json.dumps(metadata),
    )


@dataclass
class MedCPTRetriever:
    cases: list[dict[str, Any]]
    embeddings: np.ndarray
    case_ids: list[str]
    case_by_id: dict[str, dict[str, Any]]

    @classmethod
    def from_index(cls, cases_path: Path, index_path: Path) -> "MedCPTRetriever":
        cases = load_cases_jsonl(cases_path)
        case_by_id = {case["case_id"]: case for case in cases}
        data = np.load(index_path, allow_pickle=False)
        case_ids = [str(case_id) for case_id in data["case_ids"].tolist()]
        embeddings = data["embeddings"].astype("float32")
        indexed_cases = [case_by_id[case_id] for case_id in case_ids]
        return cls(
            cases=indexed_cases,
            embeddings=embeddings,
            case_ids=case_ids,
            case_by_id=case_by_id,
        )

    def search(self, query: str, top_k: int = 5, batch_size: int = 16, device: str | None = None) -> list[dict[str, Any]]:
        query_embedding = encode_queries([query], batch_size=batch_size, device=device)[0]
        scores = self.embeddings @ query_embedding
        ranked_indices = scores.argsort()[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for rank, index in enumerate(ranked_indices, start=1):
            case = self.cases[int(index)]
            results.append(
                {
                    "rank": rank,
                    "case_id": case["case_id"],
                    "score": float(scores[int(index)]),
                    "findings": case.get("findings", ""),
                    "impression": case.get("impression", ""),
                    "images": case.get("images", []),
                }
            )
        return results
