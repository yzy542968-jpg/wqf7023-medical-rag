from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Protocol


class TokenizerLike(Protocol):
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        verbose: bool,
    ) -> Mapping[str, Any]: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str: ...

    def num_special_tokens_to_add(self, pair: bool = False) -> int: ...


def normalize_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split())


def sentence_units(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
    return [piece for piece in pieces if piece]


def token_count(tokenizer: TokenizerLike, text: str) -> int:
    payload = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        verbose=False,
    )
    token_ids = payload["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def split_overlimit_text(
    tokenizer: TokenizerLike,
    text: str,
    *,
    prefix: str,
    max_tokens: int,
) -> list[str]:
    prefix_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
    special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
    content_limit = max_tokens - prefix_tokens - special_tokens
    if content_limit <= 0:
        raise ValueError("Section prefix leaves no room for report content.")

    content_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(content_ids), content_limit):
        decoded = normalize_whitespace(
            tokenizer.decode(
                content_ids[start : start + content_limit],
                skip_special_tokens=True,
            )
        )
        if not decoded:
            raise ValueError("Tokenizer decoding produced an empty over-limit segment.")
        chunk = f"{prefix}{decoded}"
        if token_count(tokenizer, chunk) > max_tokens:
            raise ValueError("Decoded over-limit segment exceeds the token budget.")
        chunks.append(chunk)
    return chunks


def pack_section_chunks(
    tokenizer: TokenizerLike,
    section: str,
    text: str,
    *,
    max_tokens: int = 64,
) -> list[dict[str, Any]]:
    prefix = f"{section.title()}: "
    units: list[str] = []
    overlimit_split_count = 0
    for sentence in sentence_units(text):
        candidate = f"{prefix}{sentence}"
        if token_count(tokenizer, candidate) <= max_tokens:
            units.append(candidate)
        else:
            split = split_overlimit_text(
                tokenizer,
                sentence,
                prefix=prefix,
                max_tokens=max_tokens,
            )
            units.extend(split)
            overlimit_split_count += len(split) - 1

    packed: list[str] = []
    for unit in units:
        unit_content = unit[len(prefix) :]
        if packed:
            combined = f"{packed[-1]} {unit_content}"
            if token_count(tokenizer, combined) <= max_tokens:
                packed[-1] = combined
                continue
        packed.append(unit)

    return [
        {
            "section": section,
            "position": position,
            "text": chunk,
            "token_count": token_count(tokenizer, chunk),
            "max_tokens": max_tokens,
            "overlimit_sentence_split_count": overlimit_split_count,
        }
        for position, chunk in enumerate(packed, start=1)
    ]


def build_report_chunks(
    case: Mapping[str, Any],
    tokenizer: TokenizerLike,
    *,
    max_tokens: int = 64,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for section in ("findings", "impression"):
        section_chunks = pack_section_chunks(
            tokenizer,
            section,
            str(case.get(section, "")),
            max_tokens=max_tokens,
        )
        for row in section_chunks:
            chunks.append(
                {
                    "case_id": str(case["case_id"]),
                    "chunk_id": (
                        f"{case['case_id']}::{section}::{int(row['position']):03d}"
                    ),
                    **row,
                }
            )
    if not chunks:
        raise ValueError(f"Case {case['case_id']} produced no report chunks.")
    return chunks
