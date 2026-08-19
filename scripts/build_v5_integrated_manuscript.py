from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / "docs/P2_V5_FRONT_MATTER_CH1_2.md"
MIDDLE = ROOT / "docs/P2_V5_INTEGRATED_CHAPTERS_3_5.md"
BACK = ROOT / "docs/P2_V5_BACK_MATTER.md"
OUTPUT = ROOT / "docs/P2_V5_INTEGRATED_MANUSCRIPT.md"


def strip_middle_header(text: str) -> str:
    marker = "# Chapter 3: Methodology"
    if marker not in text:
        raise ValueError(f"Missing expected marker in {MIDDLE}")
    return marker + text.split(marker, maxsplit=1)[1]


def main() -> None:
    parts = [
        FRONT.read_text(encoding="utf-8").strip(),
        strip_middle_header(MIDDLE.read_text(encoding="utf-8")).strip(),
        BACK.read_text(encoding="utf-8").strip(),
    ]
    manuscript = "\n\n".join(parts) + "\n"
    OUTPUT.write_text(manuscript, encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(manuscript.split())} words)")


if __name__ == "__main__":
    main()
