from __future__ import annotations

from medical_rag.multimodal.v6_chunking import build_report_chunks, token_count


class WhitespaceTokenizer:
    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}
        self._reverse: dict[int, str] = {}

    def _id(self, token: str) -> int:
        if token not in self._vocabulary:
            value = len(self._vocabulary) + 10
            self._vocabulary[token] = value
            self._reverse[value] = token
        return self._vocabulary[token]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        ids = [self._id(token) for token in text.split()]
        return [1, *ids, 2] if add_special_tokens else ids

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> dict[str, list[int]]:
        assert truncation is False
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return " ".join(self._reverse[token_id] for token_id in token_ids)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        assert pair is False
        return 2


def test_v6_chunker_preserves_section_order_and_token_budget() -> None:
    tokenizer = WhitespaceTokenizer()
    case = {
        "case_id": "CXR1",
        "findings": "One two three four. Five six seven eight.",
        "impression": "No acute disease.",
    }
    chunks = build_report_chunks(case, tokenizer, max_tokens=8)
    assert [row["section"] for row in chunks] == [
        "findings",
        "findings",
        "impression",
    ]
    assert [row["chunk_id"] for row in chunks] == [
        "CXR1::findings::001",
        "CXR1::findings::002",
        "CXR1::impression::001",
    ]
    assert all(token_count(tokenizer, row["text"]) <= 8 for row in chunks)


def test_v6_chunker_splits_overlimit_sentence_without_overlap() -> None:
    tokenizer = WhitespaceTokenizer()
    words = [f"word{index}" for index in range(15)]
    case = {
        "case_id": "CXR2",
        "findings": " ".join(words) + ".",
        "impression": "Stable finding.",
    }
    chunks = build_report_chunks(case, tokenizer, max_tokens=8)
    finding_chunks = [row for row in chunks if row["section"] == "findings"]
    reconstructed = " ".join(
        row["text"].removeprefix("Findings: ") for row in finding_chunks
    )
    assert reconstructed.split() == (" ".join(words) + ".").split()
    assert len(finding_chunks) == 3
    assert all(row["token_count"] <= 8 for row in finding_chunks)
    assert all(row["overlimit_sentence_split_count"] == 2 for row in finding_chunks)
