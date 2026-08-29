from __future__ import annotations

from pathlib import Path

import build_v10_v11_final_docx as builder


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    builder.SOURCE = ROOT / "docs" / "P2_FINAL_MANUSCRIPT.md"
    builder.OUTPUT = ROOT / "deliverables" / "22097191_ZHANG_YUE_Final_Research_Project.docx"
    builder.main()


if __name__ == "__main__":
    main()
