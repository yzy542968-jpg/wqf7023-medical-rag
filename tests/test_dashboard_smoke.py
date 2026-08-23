from __future__ import annotations

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_demo_mode_loads_and_runs_patient_scoped_pipeline(monkeypatch) -> None:
    monkeypatch.setenv("MEDICAL_RAG_DEMO_MODE", "1")
    app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "Multimodal Similar-Case Medical RAG"
    assert any("Demo Mode" in item.value for item in app.info)
    assert any(
        item.label == "Supplemental validity and robustness audits"
        for item in app.expander
    )
    assert any(
        "accepted all 24 exploratory case labels" in item.value
        for item in app.caption
    )

    app.button[0].click().run()

    assert not app.exception
    assert "pipeline_result" in app.session_state
    result = app.session_state["pipeline_result"]
    assert result["workflow"] == "v2_patient_scoped"
    assert result["model"] == "__extractive_demo__"
    assert result["retrieved_cases"]
