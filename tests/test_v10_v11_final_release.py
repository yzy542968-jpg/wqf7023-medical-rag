from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "docs" / "P2_V10_V11_FINAL_MANUSCRIPT.md"
DOCX = ROOT / "deliverables" / "22097191_ZHANG_YUE_Final_Research_Project.docx"
MANIFEST = ROOT / "artifacts" / "v10_v11_final_release_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _docx_text(path: Path) -> str:
    with ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    return "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml))


def test_final_manuscript_has_one_current_numbered_narrative() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    headings = re.findall(r"^## \d+\.\d+ .+$", text, flags=re.MULTILINE)
    duplicates = {heading: count for heading, count in Counter(headings).items() if count > 1}

    assert duplicates == {}
    assert text.count("## 1.7 Scope and Boundaries") == 1
    assert text.count("## 2.11 Similar-Case Multimodal RAG and Final Research Gap") == 1
    assert "## 2.12" not in text
    assert "## 2.13" not in text
    assert "V9 addresses these gaps" not in text
    assert "The final claims are based on the V9 held-out study" not in text
    assert "## 4.11 Post-hoc Relevance-Construct Sensitivity" in text
    assert "feature-metric coupling" in text
    assert 10_000 < len(text.split()) < 30_000
    assert "\ufffd" not in text


def test_every_parenthetical_author_year_citation_has_a_reference() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    main, remainder = text.split("# References", maxsplit=1)
    references = remainder.split("# Appendices", maxsplit=1)[0]
    citation_pairs = set(
        re.findall(r"\(([A-Z][A-Za-z-]+)(?: et al\.)?, (\d{4})\)", main)
    )
    reference_pairs = {
        (match.group(1), match.group(2))
        for match in re.finditer(r"^([A-Z][A-Za-z-]+),.*?\((\d{4})\)\.", references, re.MULTILINE)
    }
    assert citation_pairs - reference_pairs == set()


def test_historical_appendix_tables_use_appendix_numbering() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    assert "Table H.1. Retrieval results under four principal input conditions" in text
    assert "Table H.2. End-to-end QA comparison" in text
    assert "Table H.3. Runtime and computational cost" in text


def test_final_docx_matches_the_current_section_structure() -> None:
    text = _docx_text(DOCX)
    assert text.count("1.7 Scope and Boundaries") == 1
    assert text.count("2.11 Similar-Case Multimodal RAG and Final Research Gap") == 1
    assert "2.12 Similar-Case Multimodal RAG and the Final Research Gap" not in text
    assert "2.13 Final Research Gap" not in text


def test_release_manifest_hashes_all_registered_files() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for record in manifest["files"].values():
        path = ROOT / record["path"]
        assert path.is_file(), record["path"]
        assert path.stat().st_size == record["bytes"], record["path"]
        assert _sha256(path) == record["sha256"], record["path"]


def test_public_labels_do_not_misstate_the_frozen_evidence_policy() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "R5 hierarchical RAG" not in readme
    assert '"Condition": "G2 R5 hierarchical RAG"' not in app
    assert "frozen V10 evidence policy is E0 whole report" in readme
