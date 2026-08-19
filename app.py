from __future__ import annotations

import html
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

from medical_rag.agentic.evidence_checker import check_evidence_support
from medical_rag.agentic.closed_loop_agent import ClosedLoopEvidenceAgent
from medical_rag.agentic.semantic_evidence_checker import (
    MedicalNLIPredictor,
    check_semantic_evidence_support,
)
from medical_rag.agentic.planner import plan_question
from medical_rag.dashboard.demo_generation import extractive_demo_answer
from medical_rag.dashboard.multimodal_runtime import (
    answer_with_evidence_agent,
    encode_uploaded_image,
    paired_shortlist_retrieve,
)
from medical_rag.dashboard.v6_runtime import (
    V6DashboardResources,
    build_v6_generation_prompt,
    encode_uploaded_image as encode_v6_uploaded_image,
    extractive_v6_answer,
    load_v6_resources as load_v6_runtime_resources,
    retrieve_v6,
)
from medical_rag.dashboard.runtime import resolve_dashboard_runtime
from medical_rag.evaluation.case_scoped_benchmark import build_case_chunks, expected_section
from medical_rag.evaluation.answer_metrics import extract_final_answer
from medical_rag.retrieval.adaptive_retrieval import select_adaptive_top1
from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.retrieval.hybrid_retriever import HybridBM25MedCPTRetriever
from medical_rag.retrieval.medcpt_reranker import MedCPTReranker
from medical_rag.retrieval.medcpt_retriever import MedCPTRetriever
from medical_rag.retrieval.scoped_chunk_retriever import ScopedBM25ChunkRetriever
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


RUNTIME = resolve_dashboard_runtime(ROOT)
CASES_PATH = RUNTIME.cases_path
MEDCPT_INDEX_PATH = RUNTIME.dense_index_path
QA_PATH = ROOT / "data" / "processed" / "openi_case_qa_seed_clean.json"
IMAGE_ROOT = ROOT / "data" / "raw" / "images" / "images_normalized"
HYBRID_SELECTION_PATH = (
    ROOT / "experiments" / "final_optimized" / "retrieval" / "hybrid_alpha_selection.json"
)
ADAPTIVE_SELECTION_PATH = (
    ROOT
    / "experiments"
    / "final_optimized"
    / "adaptive_retrieval"
    / "adaptive_policy_selection.json"
)
SEMANTIC_SELECTION_PATH = (
    ROOT
    / "experiments"
    / "final_optimized"
    / "semantic_agent"
    / "semantic_agent_selection.json"
)
V2_TEST_SUMMARY_PATH = (
    ROOT / "experiments" / "benchmark_v2" / "final_test_evaluation" / "test_generation_summary.json"
)
V2_CONFIRMATION_SUMMARY_PATH = (
    ROOT / "experiments" / "benchmark_v2" / "confirmation_evaluation" / "test_generation_summary.json"
)
V2_CONFIRMATION_RETRIEVAL_PATH = (
    ROOT
    / "experiments"
    / "benchmark_v2"
    / "confirmation_retrieval"
    / "confirmation_retrieval_summary.json"
)
V2_VALIDITY_AUDIT_PATH = (
    ROOT
    / "experiments"
    / "benchmark_v2"
    / "validity_audit"
    / "benchmark_v2_validity_audit.json"
)
V2_VERIFIER_SELECTION_PATH = (
    ROOT
    / "experiments"
    / "benchmark_v2"
    / "calibration"
    / "semantic_verifier"
    / "semantic_agent_selection.json"
)
V2_TOP_K_SELECTION_PATH = (
    ROOT / "experiments" / "benchmark_v2" / "calibration" / "locked_top_k.json"
)
V21_SUMMARY_PATH = ROOT / "experiments" / "post_submission_v21" / "summary.json"
V21_TRANSFER_SUMMARY_PATH = (
    ROOT
    / "experiments"
    / "post_submission_v21"
    / "template_transfer"
    / "summary.json"
)
LOCKED_REPLICATION_SUMMARY_PATH = (
    ROOT / "experiments" / "locked_replication" / "summary.json"
)
V22_SUMMARY_PATH = ROOT / "experiments" / "post_submission_v22" / "summary.json"
V23_SUMMARY_PATH = ROOT / "experiments" / "post_submission_v23" / "summary.json"
MULTIMODAL_V42_CONFIG_PATH = ROOT / "config" / "multimodal_v42.json"
MULTIMODAL_V42_CACHE_PATH = (
    ROOT / "data" / "processed" / "multimodal_v41_biovil_t_embeddings.npz"
)
MULTIMODAL_V42_SUMMARY_PATH = (
    ROOT / "experiments" / "post_submission_v42" / "confirmation_retrieval_summary.json"
)
MULTIMODAL_V42_STATISTICS_PATH = (
    ROOT / "experiments" / "post_submission_v42" / "confirmation_statistics.json"
)
MULTIMODAL_V42_RUNTIME_PATH = (
    ROOT / "experiments" / "post_submission_v42" / "runtime_profile.json"
)
V6_CONFIG_PATH = ROOT / "config" / "v6_confirmation.json"
V6_COHORT_PATH = ROOT / "data" / "splits" / "v6" / "v6_confirmation_cohort.json"
V6_MEDSIGLIP_CACHE_PATH = (
    ROOT / "data" / "processed" / "v6_confirmation_medsiglip_embeddings.npz"
)
MODEL_OPTIONS = {
    "Qwen2.5-1.5B (full experiment)": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-0.5B (faster demo)": "Qwen/Qwen2.5-0.5B-Instruct",
}
DEMO_MODEL_OPTIONS = {"Extractive demo (no model download)": "__extractive_demo__"}
PROMPT_OPTIONS = {
    "Direct": "direct",
    "Evidence-guided": "evidence_guided",
    "Structured case-aware": "structured_case_aware",
}
RETRIEVER_OPTIONS = {
    "Adaptive Hybrid + reranker": "adaptive",
    "Locked Hybrid (alpha=0.30)": "hybrid",
    "BM25": "bm25",
}
AGENT_OPTIONS = {
    "Hybrid Medical NLI": "semantic",
    "Lexical + negation rules": "rule",
    "Disabled": "off",
}
V2_TASKS = {
    "Findings": (
        "case_scoped_findings",
        "What radiographic findings are documented for this examination?",
    ),
    "Impression": (
        "case_scoped_impression",
        "What is the final radiology impression for this examination?",
    ),
    "Report summary": (
        "case_scoped_summary",
        "Summarize the principal abnormality or conclusion in this report.",
    ),
}


st.set_page_config(
    page_title="Evidence-Checking Medical RAG",
    page_icon=":material/radiology:",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1500px; padding-top: 1.4rem; padding-bottom: 3rem;}
    h1 {font-size: 2rem !important; font-weight: 680 !important; letter-spacing: 0 !important;}
    h2, h3 {letter-spacing: 0 !important;}
    div[data-testid="stMetric"] {border-left: 3px solid #087E8B; padding-left: 0.8rem;}
    div[data-testid="stMetricLabel"] {font-size: 0.78rem; color: #54636C;}
    div[data-testid="stAlert"] {border-radius: 6px;}
    .evidence-label {font-size: 0.75rem; font-weight: 700; color: #54636C; text-transform: uppercase;}
    .answer-band {border-left: 4px solid #087E8B; background: #FFFFFF; padding: 1rem 1.1rem; margin: 0.4rem 0 1rem;}
    .research-note {border-left: 4px solid #C65D34; background: #FFF8F4; padding: 0.8rem 1rem; color: #5A3324;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_bm25_resources() -> tuple[list[dict[str, Any]], BM25Retriever]:
    cases = load_cases_jsonl(CASES_PATH)
    return cases, BM25Retriever().fit(cases)


@st.cache_resource(show_spinner=False)
def load_hybrid_resources() -> tuple[HybridBM25MedCPTRetriever, Any, Any, str]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    cases, bm25 = load_bm25_resources()
    if MEDCPT_INDEX_PATH is None:
        raise RuntimeError("Dense retrieval is unavailable in Demo Mode.")
    medcpt = MedCPTRetriever.from_index(CASES_PATH, MEDCPT_INDEX_PATH)
    selection = json.loads(HYBRID_SELECTION_PATH.read_text(encoding="utf-8"))
    hybrid = HybridBM25MedCPTRetriever.from_components(
        cases, bm25, medcpt, alpha=float(selection["selected_alpha"])
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
    model.eval()
    return hybrid, tokenizer, model, device


@st.cache_resource(show_spinner=False)
def load_reranker() -> MedCPTReranker:
    return MedCPTReranker(local_files_only=True, batch_size=8)


@st.cache_resource(show_spinner=False)
def load_semantic_predictor() -> MedicalNLIPredictor:
    selection = json.loads(SEMANTIC_SELECTION_PATH.read_text(encoding="utf-8"))
    return MedicalNLIPredictor(
        selection["nli_model"], local_files_only=True, batch_size=16
    )


@st.cache_resource(show_spinner=False)
def load_scoped_resources() -> tuple[dict[str, dict[str, Any]], ScopedBM25ChunkRetriever]:
    cases, _ = load_bm25_resources()
    case_by_id = {str(case["case_id"]): case for case in cases}
    chunks = [chunk for case in cases for chunk in build_case_chunks(case)]
    return case_by_id, ScopedBM25ChunkRetriever().fit(chunks)


@st.cache_data(show_spinner=False)
def load_v2_configs() -> tuple[int, dict[str, Any]]:
    top_k = json.loads(V2_TOP_K_SELECTION_PATH.read_text(encoding="utf-8"))["selected_top_k"]
    verifier = json.loads(V2_VERIFIER_SELECTION_PATH.read_text(encoding="utf-8"))["selected_config"]
    return int(top_k), verifier


@st.cache_data(show_spinner=False)
def load_v21_answerability_threshold() -> float:
    payload = json.loads(V21_SUMMARY_PATH.read_text(encoding="utf-8"))
    return float(
        payload["threshold_selection"]["closed_loop_agent_v2"]["threshold"]
    )


@st.cache_data(show_spinner=False)
def load_locked_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    adaptive = json.loads(ADAPTIVE_SELECTION_PATH.read_text(encoding="utf-8"))
    semantic = json.loads(SEMANTIC_SELECTION_PATH.read_text(encoding="utf-8"))
    return adaptive["selected_policy"], semantic["selected_config"]


def run_multimodal_semantic_check(answer: str, evidence: str) -> Any:
    _, semantic_config = load_locked_configs()
    return check_semantic_evidence_support(
        answer,
        evidence,
        load_semantic_predictor(),
        min_combined_support=float(semantic_config["support_threshold"]),
        entailment_threshold=float(semantic_config["entailment_threshold"]),
        contradiction_threshold=float(semantic_config["contradiction_threshold"]),
        lexical_weight=float(semantic_config["lexical_weight"]),
    )


@st.cache_resource(show_spinner=False)
def load_generator(model_name: str) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        local_files_only=True,
    ).to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource(show_spinner=False)
def load_multimodal_resources() -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    list[str],
    BM25Retriever,
    np.ndarray,
    Any,
]:
    if RUNTIME.is_demo:
        raise RuntimeError("The paired image workflow requires the local full OpenI artifacts.")
    from medical_rag.multimodal.biovilt import BioVilTEncoder

    config = json.loads(MULTIMODAL_V42_CONFIG_PATH.read_text(encoding="utf-8"))
    all_cases = {str(case["case_id"]): case for case in load_cases_jsonl(CASES_PATH)}
    candidate_ids = set()
    for split in ("development", "confirmation"):
        benchmark = json.loads(
            (ROOT / config["cohorts"][split]["benchmark_path"]).read_text(encoding="utf-8")
        )
        candidate_ids.update(str(row["case_id"]) for row in benchmark["questions"])
    ordered_ids = sorted(candidate_ids)
    cases = {case_id: all_cases[case_id] for case_id in ordered_ids}
    cache = np.load(MULTIMODAL_V42_CACHE_PATH, allow_pickle=False)
    if cache["case_ids"].tolist() != ordered_ids:
        raise RuntimeError("Multimodal embedding cache does not match the locked candidate pool.")
    report_embeddings = np.asarray(cache["report_embeddings"], dtype=np.float32)
    bm25 = BM25Retriever().fit([cases[case_id] for case_id in ordered_ids])
    encoder = BioVilTEncoder(
        model_name=config["encoder"]["joint_encoder"],
        text_revision=config["encoder"]["text_model_revision"],
        device="cuda",
        text_max_length=int(config["encoder"]["text_max_length"]),
    )
    return config, cases, ordered_ids, bm25, report_embeddings, encoder


@st.cache_resource(show_spinner=False)
def load_v6_dashboard_resources() -> V6DashboardResources:
    return load_v6_runtime_resources(
        config_path=V6_CONFIG_PATH,
        cohort_path=V6_COHORT_PATH,
        cases_path=CASES_PATH,
        medsiglip_cache_path=V6_MEDSIGLIP_CACHE_PATH,
    )


@st.cache_resource(show_spinner=False)
def load_v6_image_encoder() -> Any:
    import torch

    from medical_rag.multimodal.medsiglip import MedSiglipEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = json.loads(V6_CONFIG_PATH.read_text(encoding="utf-8"))
    encoder_config = config["multimodal_retrieval"]["primary_encoder"]
    return MedSiglipEncoder(
        model_name=str(encoder_config["model"]),
        revision=str(encoder_config["revision"]),
        device=device,
        cache_dir=ROOT / ".hf_cache",
        max_text_tokens=int(encoder_config["max_text_tokens"]),
        local_files_only=True,
    )


@st.cache_data(show_spinner=False)
def load_examples() -> list[dict[str, Any]]:
    payload = json.loads(QA_PATH.read_text(encoding="utf-8"))
    return payload["questions"]


def encode_medcpt_query(query: str, tokenizer: Any, model: Any, device: str) -> np.ndarray:
    import torch

    encoded = tokenizer(
        [query], truncation=True, padding=True, return_tensors="pt", max_length=64
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        embedding = model(**encoded).last_hidden_state[:, 0, :]
    embedding = embedding.detach().cpu().numpy().astype("float32")
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return embedding[0] / norm[0]


def retrieve(question: str, method: str, top_k: int) -> list[dict[str, Any]]:
    if method == "bm25":
        _, bm25 = load_bm25_resources()
        return bm25.search(question, top_k=top_k)
    hybrid, tokenizer, model, device = load_hybrid_resources()
    embedding = encode_medcpt_query(question, tokenizer, model, device)
    return hybrid.search_with_embedding(question, embedding, top_k=top_k)


def retrieve_with_policy(
    question: str, method: str, top_k: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if method != "adaptive":
        results = retrieve(question, method, top_k)
        selected = results[:1]
        return results, selected, {
            "source": method,
            "abstained": not selected,
            "reason": "fixed_retriever",
            "base_margin": None,
            "reranker_margin": None,
        }

    candidate_depth = max(3, top_k)
    base_results = retrieve(question, "hybrid", candidate_depth)
    reranked = load_reranker().rerank(question, base_results[:3])
    policy, _ = load_locked_configs()
    decision = select_adaptive_top1(
        base_case_ids=[str(item["case_id"]) for item in base_results[:3]],
        base_scores=[float(item["score"]) for item in base_results[:3]],
        reranked_case_ids=[str(item["case_id"]) for item in reranked],
        reranker_scores=[float(item["reranker_score"]) for item in reranked],
        reranker_margin_threshold=float(policy["reranker_margin_threshold"]),
        base_margin_threshold=float(policy["base_margin_threshold"]),
        minimum_base_score=float(policy["minimum_base_score"]),
        minimum_selected_margin=float(policy["minimum_selected_margin"]),
    )
    reranker_scores = {
        str(item["case_id"]): float(item["reranker_score"]) for item in reranked
    }
    display_results = []
    for rank, item in enumerate(base_results[:top_k], start=1):
        enriched = {
            **item,
            "rank": rank,
            "reranker_score": reranker_scores.get(str(item["case_id"])),
            "selected": str(item["case_id"]) == decision.selected_case_id,
        }
        display_results.append(enriched)
    selected = [
        item for item in display_results if str(item["case_id"]) == decision.selected_case_id
    ]
    return display_results, selected, asdict(decision)


def evidence_context(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No sufficiently confident case was retrieved."
    return "\n\n".join(
        "\n".join(
            [
                f"Case ID: {case['case_id']}",
                f"Findings: {case.get('findings', '')}",
                f"Impression: {case.get('impression', '')}",
            ]
        )
        for case in results
    )


def build_prompt(question: str, context: str, mode: str) -> str:
    common = [
        "Question:",
        question,
        "",
        "Selected radiology case:",
        context,
        "",
    ]
    if mode == "direct":
        return "\n".join(
            [
                "Answer the medical question using the selected radiology case.",
                *common,
                "Answer clearly and concisely.",
            ]
        )
    if mode == "structured_case_aware":
        return "\n".join(
            [
                "Answer this case-grounded research question using only the selected evidence.",
                "Do not combine facts from other patients or add outside clinical knowledge.",
                "If the evidence is insufficient, state that it is insufficient.",
                *common,
                "Respond in this structure:",
                "Evidence: one short statement with the selected Case ID.",
                "Final answer: one concise paragraph supported by that case.",
            ]
        )
    return "\n".join(
        [
            "Answer the medical question using only the selected radiology case evidence.",
            "Do not add unsupported findings, diagnoses, locations, or severity.",
            "If the evidence is insufficient, state that it is insufficient.",
            *common,
            "Return only one concise answer paragraph.",
        ]
    )


def generate(prompt: str, model_name: str) -> tuple[str, str]:
    if model_name == "__extractive_demo__":
        answer = extractive_demo_answer(prompt)
        return answer, answer
    import torch

    tokenizer, model, device = load_generator(model_name)
    messages = [
        {
            "role": "system",
            "content": "You are a careful report-grounded medical QA assistant for a research prototype.",
        },
        {"role": "user", "content": prompt},
    ]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return raw, extract_final_answer(raw)


def parse_uploaded_question(uploaded_file: Any) -> tuple[str, str | None]:
    if uploaded_file is None:
        return "", None
    text = uploaded_file.getvalue().decode("utf-8-sig").strip()
    if uploaded_file.name.lower().endswith(".json"):
        payload = json.loads(text)
        return str(payload.get("question", "")).strip(), payload.get("question_type")
    return text, None


def parse_uploaded_scoped_request(uploaded_file: Any) -> tuple[str, str | None, str | None]:
    if uploaded_file is None:
        return "", None, None
    text = uploaded_file.getvalue().decode("utf-8-sig").strip()
    if uploaded_file.name.lower().endswith(".json"):
        payload = json.loads(text)
        return (
            str(payload.get("question", "")).strip(),
            payload.get("question_type"),
            str(payload.get("case_id", "")).strip() or None,
        )
    return text, None, None


def retrieve_scoped_evidence(
    case_id: str, question: str, question_type: str, top_k: int
) -> list[dict[str, Any]]:
    case_by_id, retriever = load_scoped_resources()
    if case_id not in case_by_id:
        raise ValueError(f"Unknown patient case ID: {case_id}")
    return retriever.search(
        question,
        top_k=top_k,
        case_id=case_id,
        allowed_sections={expected_section(question_type)},
    )


def build_scoped_live_prompt(case_id: str, question: str, evidence: list[dict[str, Any]]) -> str:
    evidence_text = "\n".join(
        f"[{row['section']} {row['position']}] {row['text']}" for row in evidence
    )
    return "\n".join(
        [
            "Answer the question using only the retrieved evidence from the specified radiology case.",
            "Do not add findings that are absent from the evidence.",
            f"Case scope: {case_id}",
            f"Question: {question}",
            "",
            "Retrieved evidence:",
            evidence_text,
            "",
            "Answer clearly and concisely.",
        ]
    )


def local_image_paths(case: dict[str, Any]) -> list[Path]:
    paths = []
    for image in case.get("images", []):
        filename = Path(str(image.get("filename", ""))).name
        path = IMAGE_ROOT / filename
        if path.exists():
            paths.append(path)
    return paths


def render_pipeline_result(result: dict[str, Any]) -> None:
    checks = result["sentence_checks"]
    retrieval = result["retrieval_decision"]
    metric_columns = st.columns(5)
    metric_columns[0].metric("Retrieved", len(result["retrieved_cases"]))
    top_score = result["retrieved_cases"][0]["score"] if result["retrieved_cases"] else 0.0
    metric_columns[1].metric("Top-1 score", f"{top_score:.3f}")
    metric_columns[2].metric("Evidence support", f"{result['support_rate']:.1%}")
    agent_state = (
        "Advisory"
        if result.get("agent_action_policy") == "audit_only"
        else (
            "Abstained"
            if result["abstained"]
            else ("Answered" if result.get("planning_trace") else "Filtered")
        )
    )
    metric_columns[3].metric("Agent", agent_state)
    metric_columns[4].metric("Latency", f"{result['latency_seconds']:.1f}s")

    source_labels = {
        "agreement": "Hybrid and reranker agree",
        "reranker": "Reranker selected",
        "hybrid": "Hybrid retained",
        "bm25": "BM25 fixed ranking",
        "patient_scope": "Patient scope and section route",
        "closed_loop_agent": "Closed-loop inferred route",
    }
    retrieval_message = source_labels.get(retrieval["source"], retrieval["source"])
    if retrieval["abstained"]:
        st.warning(f"Retrieval abstained: {retrieval['reason']}")
    else:
        st.caption(f"Retrieval decision: {retrieval_message} · {retrieval['reason']}")

    st.subheader("Grounded answer")
    answer_class = "research-note" if result["abstained"] else "answer-band"
    st.markdown(
        f'<div class="{answer_class}">{result["final_answer"]}</div>',
        unsafe_allow_html=True,
    )

    draft_col, trace_col = st.columns([1, 1], gap="large")
    with draft_col:
        st.subheader("Generation")
        st.text_area("Draft answer", result["draft_answer"], height=180, disabled=True)
        with st.expander("Prompt", icon=":material/article:"):
            st.code(result["prompt"], language="text")
    with trace_col:
        st.subheader("Agent trace")
        if result.get("planning_trace"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Step": row["step"],
                            "Action": row["action"],
                            "Intent": row["intent"],
                            "Evidence score": row["evidence_score"],
                            "Reason": row["reason"],
                        }
                        for row in result["planning_trace"]
                    ]
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "Evidence score": st.column_config.ProgressColumn(
                        min_value=0, max_value=1
                    )
                },
            )
        if checks:
            trace_rows = [
                {
                    "Decision": (
                        ("Supported" if check["supported"] else "Review")
                        if result.get("agent_action_policy") == "audit_only"
                        else ("Keep" if check["supported"] else "Remove")
                    ),
                    "Score": round(
                        check.get("combined_support_score", check.get("support_score", 0.0)),
                        3,
                    ),
                    "Entailment": (
                        round(check["entailment_probability"], 3)
                        if "entailment_probability" in check
                        else None
                    ),
                    "Contradiction": (
                        round(check["contradiction_probability"], 3)
                        if "contradiction_probability" in check
                        else None
                    ),
                    "Polarity": "Aligned" if check["negation_consistent"] else "Conflict",
                    "Reason": check.get("decision_reason", "lexical_rule"),
                    "Answer sentence": check["sentence"],
                    "Matched evidence": check["matched_evidence"],
                }
                for check in checks
            ]
            st.dataframe(
                pd.DataFrame(trace_rows),
                width="stretch",
                hide_index=True,
                column_config={"Score": st.column_config.ProgressColumn(min_value=0, max_value=1)},
            )
        elif not result.get("planning_trace"):
            st.info("No answer sentence was available for checking.")

    if result.get("workflow") == "v2_patient_scoped":
        st.subheader("Retrieved evidence")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Rank": row["rank"],
                        "Case": row["case_id"],
                        "Section": row["section"],
                        "Score": row["score"],
                        "Evidence": row["text"],
                    }
                    for row in result["retrieved_cases"]
                ]
            ),
            width="stretch",
            hide_index=True,
            column_config={"Score": st.column_config.NumberColumn(format="%.3f")},
        )
        case = result.get("scoped_case", {})
        with st.expander("Patient-scoped source report", icon=":material/description:"):
            st.markdown('<div class="evidence-label">Findings</div>', unsafe_allow_html=True)
            st.write(case.get("findings") or "Not reported")
            st.markdown('<div class="evidence-label">Impression</div>', unsafe_allow_html=True)
            st.write(case.get("impression") or "Not reported")
        export = {key: value for key, value in result.items() if key != "prompt"}
        st.download_button(
            "Export run",
            data=json.dumps(export, indent=2, ensure_ascii=False),
            file_name="medical_rag_v2_run.json",
            mime="application/json",
            icon=":material/download:",
        )
        return

    st.subheader("Retrieved cases")
    for case in result["retrieved_cases"]:
        score_parts = [f"score {case['score']:.3f}"]
        if "bm25_score" in case:
            score_parts.append(f"BM25 {case['bm25_score']:.3f}")
            score_parts.append(f"MedCPT {case['medcpt_score']:.3f}")
        if case.get("reranker_score") is not None:
            score_parts.append(f"reranker {case['reranker_score']:.3f}")
        selected_marker = "SELECTED · " if case.get("selected") else ""
        with st.expander(
            f"{selected_marker}#{case['rank']}  {case['case_id']}  |  {' · '.join(score_parts)}",
            expanded=bool(case.get("selected", case["rank"] == 1)),
            icon=":material/description:",
        ):
            report_col, image_col = st.columns([3, 2])
            with report_col:
                st.markdown('<div class="evidence-label">Findings</div>', unsafe_allow_html=True)
                st.write(case.get("findings") or "Not reported")
                st.markdown('<div class="evidence-label">Impression</div>', unsafe_allow_html=True)
                st.write(case.get("impression") or "Not reported")
            with image_col:
                image_paths = local_image_paths(case)
                if image_paths:
                    st.image([str(path) for path in image_paths], width=250)
                else:
                    image_names = [image.get("filename", "") for image in case.get("images", [])]
                    st.caption("Linked images: " + ", ".join(image_names))

    export = {key: value for key, value in result.items() if key != "prompt"}
    st.download_button(
        "Export run",
        data=json.dumps(export, indent=2, ensure_ascii=False),
        file_name="medical_rag_run.json",
        mime="application/json",
        icon=":material/download:",
    )


def render_results() -> None:
    final_summary = json.loads(
        (
            ROOT
            / "experiments"
            / "final_optimized"
            / "final_test"
            / "final_optimized_test_summary.json"
        ).read_text(encoding="utf-8")
    )
    alpha_selection = json.loads(HYBRID_SELECTION_PATH.read_text(encoding="utf-8"))
    adaptive_selection = json.loads(ADAPTIVE_SELECTION_PATH.read_text(encoding="utf-8"))
    contamination = json.loads(
        (
            ROOT
            / "experiments"
            / "final_optimized"
            / "contamination"
            / "report_rag_cross_case_contamination.json"
        ).read_text(encoding="utf-8")
    )
    validity = json.loads(
        (
            ROOT
            / "experiments"
            / "final_optimized"
            / "validity_audit"
            / "research_validity_audit.json"
        ).read_text(encoding="utf-8")
    )
    v2_test = json.loads(V2_TEST_SUMMARY_PATH.read_text(encoding="utf-8"))
    v2_confirmation = json.loads(V2_CONFIRMATION_SUMMARY_PATH.read_text(encoding="utf-8"))
    v2_retrieval = json.loads(V2_CONFIRMATION_RETRIEVAL_PATH.read_text(encoding="utf-8"))
    v2_verifier = json.loads(V2_VERIFIER_SELECTION_PATH.read_text(encoding="utf-8"))
    v2_validity = json.loads(V2_VALIDITY_AUDIT_PATH.read_text(encoding="utf-8"))
    v21 = json.loads(V21_SUMMARY_PATH.read_text(encoding="utf-8"))
    v21_transfer = json.loads(
        V21_TRANSFER_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    replication = json.loads(
        LOCKED_REPLICATION_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    v22 = json.loads(V22_SUMMARY_PATH.read_text(encoding="utf-8"))
    v23 = json.loads(V23_SUMMARY_PATH.read_text(encoding="utf-8"))

    st.subheader("Locked held-out test")
    metrics = st.columns(5)
    metrics[0].metric("Cases", "36")
    metrics[1].metric("Questions", str(final_summary["n"]))
    metrics[2].metric("Verified Token-F1", f"{final_summary['verified_token_f1']:.3f}")
    metrics[3].metric("Evidence support", f"{final_summary['evidence_support_rate']:.1%}")
    metrics[4].metric("Final abstention", f"{final_summary['final_abstention_rate']:.1%}")
    st.caption(
        "All configuration choices were made on 84 development cases; these 36 cases were "
        "kept disjoint and evaluated once."
    )

    statistics = json.loads(
        (
            ROOT
            / "experiments"
            / "final_optimized"
            / "statistics"
            / "held_out_test_grouped_bootstrap.json"
        ).read_text(encoding="utf-8")
    )
    labels = {
        "llm_only": "LLM only",
        "report_bm25_semantic_agent": "Report-RAG + semantic checker",
        "case_bm25_top1_semantic_agent": "Case BM25 + semantic checker",
        "case_hybrid_top1_a050_semantic_agent": "Previous Hybrid + semantic checker",
        "final_adaptive_direct_semantic_agent": "Final adaptive system",
    }
    summary_by_system = {item["system"]: item for item in statistics["summary"]}
    rows = []
    for system, label in labels.items():
        item = summary_by_system[system]
        rows.append(
            {
                "System": label,
                "Token-F1": item["mean_token_f1"],
                "Grouped 95% CI": (
                    f"[{item['ci_low_95']:.3f}, {item['ci_high_95']:.3f}]"
                ),
            }
        )
    result_frame = pd.DataFrame(rows)
    st.dataframe(
        result_frame,
        width="stretch",
        hide_index=True,
        column_config={"Token-F1": st.column_config.NumberColumn(format="%.3f")},
    )
    st.bar_chart(
        result_frame.set_index("System")[["Token-F1"]],
        horizontal=True,
        color="#087E8B",
    )

    pairwise = next(
        item
        for item in statistics["pairwise"]
        if item["system_a"] == "final_adaptive_direct_semantic_agent"
        and item["system_b"] == "case_bm25_top1_semantic_agent"
    )
    st.info(
        "Final vs Case BM25: "
        f"Δ Token-F1 {pairwise['mean_difference']:+.3f}, "
        f"95% CI [{pairwise['ci_low']:.3f}, {pairwise['ci_high']:.3f}], "
        f"paired randomization p={pairwise['paired_randomization_p']:.4f}, "
        f"Holm-adjusted p={pairwise['holm_adjusted_randomization_p']:.4f}."
    )

    st.subheader("Retrieval safety")
    retrieval_metrics = st.columns(4)
    retrieval_metrics[0].metric("Locked hybrid alpha", f"{alpha_selection['selected_alpha']:.2f}")
    retrieval_metrics[1].metric(
        "Hybrid test Hit@1", f"{alpha_selection['held_out_test_metrics']['hit@1']:.1%}"
    )
    retrieval_metrics[2].metric(
        "Adaptive selective accuracy",
        f"{adaptive_selection['held_out_test']['selective_accuracy']:.1%}",
    )
    retrieval_metrics[3].metric(
        "Retrieval abstention", f"{adaptive_selection['held_out_test']['abstention_rate']:.1%}"
    )
    st.warning(
        "Automated detector estimate for Report-RAG top-5 cross-case support: "
        f"{contamination['lexically_anchored_cross_case_contaminated_sentence_rate']:.1%}-"
        f"{contamination['cross_case_contaminated_sentence_rate']:.1%} of answer sentences and "
        f"{contamination['lexically_anchored_answer_cross_case_contamination_rate']:.1%}-"
        f"{contamination['answer_cross_case_contamination_rate']:.1%} of answers. "
        "Human confirmation remains required; the final system structurally exposes one case."
    )

    st.subheader("Validity and headroom audit")
    audit_metrics = st.columns(4)
    audit_metrics[0].metric(
        "Oracle verified F1",
        f"{validity['oracle_retrieval_headroom']['oracle_verified_token_f1']:.3f}",
    )
    audit_metrics[1].metric(
        "Actual-to-oracle gap",
        f"{validity['oracle_retrieval_headroom']['absolute_gap']:.3f}",
    )
    audit_metrics[2].metric(
        "Ambiguous test queries",
        f"{validity['benchmark_ambiguity']['held_out_test']['ambiguous_question_rate']:.1%}",
    )
    audit_metrics[3].metric(
        "Wrong-case support",
        f"{validity['verification_conditioned_on_retrieval']['wrong_retrieval']['evidence_support_rate']:.1%}",
    )
    st.caption(
        "The verifier measures faithfulness to retrieved evidence, not whether the correct patient was "
        "retrieved. Images are linked for display; the modeled pipeline is text-only."
    )

    st.subheader("Benchmark V2: patient-known evidence QA")
    v2_metrics = st.columns(5)
    v2_metrics[0].metric("Confirmation cases", str(v2_confirmation["case_count"]))
    v2_metrics[1].metric("Locked top-k", str(v2_confirmation["top_k"]))
    v2_metrics[2].metric(
        "Extractive Token-F1",
        f"{v2_validity['confirmation']['extractive_retrieved_context_token_f1']:.3f}",
    )
    v2_metrics[3].metric("Qwen Token-F1", f"{v2_confirmation['verified_token_f1']:.3f}")
    v2_metrics[4].metric("Evidence recall", f"{v2_confirmation['mean_retrieval_recall']:.1%}")

    retrieval_labels = {
        "global_bm25": "Global BM25",
        "case_scoped_bm25": "Patient-scoped BM25",
        "case_scoped_agent_routed_bm25": "Patient scope + planner route",
    }
    v2_retrieval_frame = pd.DataFrame(
        [
            {
                "Retrieval condition": retrieval_labels[system],
                "Hit@1": values["hit@1"],
                "Recall@5": values["recall@5"],
                "MRR": values["mrr"],
            }
            for system, values in v2_retrieval["systems"].items()
        ]
    )
    st.dataframe(
        v2_retrieval_frame,
        width="stretch",
        hide_index=True,
        column_config={
            "Hit@1": st.column_config.NumberColumn(format="%.3f"),
            "Recall@5": st.column_config.NumberColumn(format="%.3f"),
            "MRR": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.warning(
        "Controlled-workflow boundary: routed candidates equal the relevance set for "
        f"{v2_validity['confirmation']['routed_candidate_pool_equals_qrels_rate']:.0%} of confirmation "
        "queries, so routed Hit@1 is not a semantic-retrieval result. Returning retrieved context "
        f"scores {v2_validity['confirmation']['extractive_retrieved_context_token_f1']:.3f}, "
        f"versus {v2_confirmation['verified_token_f1']:.3f} for Qwen."
    )
    st.info(
        f"Diagnostic V2 test Qwen F1 {v2_test['verified_token_f1']:.3f}; primary confirmation "
        f"Qwen F1 {v2_confirmation['verified_token_f1']:.3f}, case-bootstrap 95% CI "
        f"[{v2_confirmation['verified_token_f1_case_bootstrap_95_ci'][0]:.3f}, "
        f"{v2_confirmation['verified_token_f1_case_bootstrap_95_ci'][1]:.3f}]. Calibration selected "
        f"{v2_verifier['selected_config']['action_policy'].replace('_', ' ')} verification: "
        "NLI reports grounding risk but does not automatically delete answer sentences."
    )
    st.caption(
        "V1 is an open-corpus stress test; V2 uses an explicit case-ID metadata filter. "
        "No authentication or clinical access-control layer is implemented. Their Token-F1 "
        "values are not a paired comparison, and section routing is deterministic."
    )

    st.subheader("Post-submission V2.1: closed-loop evidence Agent")
    v21_labels = {
        "fixed_report_bm25": "Fixed report BM25",
        "route_only_agent": "Route-only ablation",
        "closed_loop_agent_v2": "Closed-loop Agent",
    }
    v21_frame = pd.DataFrame(
        [
            {
                "System": label,
                "Macro F1": v21["systems"][system]["test"]["macro_f1"],
                "False-answer rate": v21["systems"][system]["test"][
                    "false_answer_rate"
                ],
                "Evidence hit rate": v21["systems"][system]["test"][
                    "retrieval_hit_rate_answerable"
                ],
                "Mean chunks": v21["systems"][system]["test"][
                    "mean_retrieved_chunks"
                ],
            }
            for system, label in v21_labels.items()
        ]
    )
    st.dataframe(
        v21_frame,
        width="stretch",
        hide_index=True,
        column_config={
            "Macro F1": st.column_config.NumberColumn(format="%.3f"),
            "False-answer rate": st.column_config.NumberColumn(format="%.1%%"),
            "Evidence hit rate": st.column_config.NumberColumn(format="%.1%%"),
            "Mean chunks": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    closed_test = v21["systems"]["closed_loop_agent_v2"]["test"]
    route_test = v21["systems"]["route_only_agent"]["test"]
    calibrated_test = v21["posthoc_probability_calibration"][
        "closed_loop_agent_v2"
    ]["splits"]["test"]
    transfer_test = v21_transfer["transfer_wording_test"]
    v21_metrics = st.columns(5)
    v21_metrics[0].metric("Cases", str(v21["case_count"]))
    v21_metrics[1].metric("Questions", str(v21["question_count"]))
    v21_metrics[2].metric(
        "Loop gain vs route-only",
        f"{closed_test['macro_f1'] - route_test['macro_f1']:+.3f}",
    )
    v21_metrics[3].metric(
        "Calibrated ECE",
        f"{calibrated_test['answerability_calibration']['ece']:.3f}",
    )
    v21_metrics[4].metric(
        "Transfer Macro F1", f"{transfer_test['macro_f1']:.3f}"
    )
    st.warning(
        "Wording-transfer stress test: Macro F1 falls from "
        f"{closed_test['macro_f1']:.3f} to {transfer_test['macro_f1']:.3f}. "
        "The deterministic planner is auditable but lexically brittle; this is a measured "
        "limitation, not a general clinical-language claim."
    )
    semantic_original = v22["original_test"]["raw"]
    semantic_transfer = v22["transfer_test"]["raw"]
    st.info(
        "Exploratory V2.2 constrained Qwen planner: original-wording Macro F1 "
        f"{semantic_original['macro_f1']:.3f}, transfer-wording Macro F1 "
        f"{semantic_transfer['macro_f1']:.3f}. It improves paraphrase transfer but "
        "weakens rejection of missing near-domain facts, so it is reported as a trade-off "
        "rather than replacing the frozen rule planner."
    )
    st.subheader("Preregistered V2.3: lexical-first semantic fallback")
    v23_rows = []
    v23_labels = {
        "original_test": "Original wording",
        "reserved_wording_set_1": "Reserved wording 1",
        "reserved_wording_set_2": "Reserved wording 2",
    }
    for set_name, label in v23_labels.items():
        result = v23["evaluation_sets"][set_name]
        for system in ("lexical", "hybrid"):
            metrics_row = result["systems"][system]["raw"]
            v23_rows.append(
                {
                    "Evaluation": label,
                    "System": "Lexical" if system == "lexical" else "Hybrid",
                    "Macro F1": metrics_row["macro_f1"],
                    "False-answer rate": metrics_row["false_answer_rate"],
                    "Evidence hit rate": metrics_row["retrieval_hit_rate_answerable"],
                    "Semantic call rate": (
                        0.0
                        if system == "lexical"
                        else result["hybrid_policy_usage"][
                            "semantic_planner_call_rate"
                        ]
                    ),
                }
            )
    st.dataframe(
        pd.DataFrame(v23_rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Macro F1": st.column_config.NumberColumn(format="%.3f"),
            "False-answer rate": st.column_config.NumberColumn(format="%.1%%"),
            "Evidence hit rate": st.column_config.NumberColumn(format="%.1%%"),
            "Semantic call rate": st.column_config.NumberColumn(format="%.1%%"),
        },
    )
    transfer2_bootstrap = v23["evaluation_sets"]["reserved_wording_set_2"][
        "paired_case_bootstrap"
    ]
    macro_delta = transfer2_bootstrap["macro_f1_delta_hybrid_minus_lexical"]
    false_delta = transfer2_bootstrap[
        "false_answer_rate_delta_hybrid_minus_lexical"
    ]
    st.warning(
        "On the second result-blind wording set, hybrid routing improves Macro F1 by "
        f"{macro_delta['observed']:+.3f} (case-bootstrap 95% CI "
        f"[{macro_delta['ci95'][0]:+.3f}, {macro_delta['ci95'][1]:+.3f}]) but also "
        f"raises false-answer rate by {false_delta['observed']:+.3f} (95% CI "
        f"[{false_delta['ci95'][0]:+.3f}, {false_delta['ci95'][1]:+.3f}]). "
        "The policy is robust but not safety-dominant under wording shift."
    )
    runtime = v23["runtime_profile"]
    cuda_runtime = runtime["cuda"]
    runtime_metrics = st.columns(4)
    runtime_metrics[0].metric(
        "Model load", f"{runtime['timing_seconds']['model_and_tokenizer_load']:.2f} s"
    )
    runtime_metrics[1].metric(
        "432 planner prompts", f"{runtime['timing_seconds']['generation']:.2f} s"
    )
    runtime_metrics[2].metric(
        "Generation throughput",
        f"{runtime['throughput_records_per_second']['generation_only']:.1f}/s",
    )
    runtime_metrics[3].metric(
        "Peak CUDA allocated", f"{cuda_runtime['peak_allocated_mib']:.0f} MiB"
    )
    st.caption(
        "Single local CUDA run on "
        f"{cuda_runtime['name']}; peak reserved {cuda_runtime['peak_reserved_mib']:.0f} MiB. "
        "Timing includes cached local model files and is descriptive, not a deployment SLA."
    )

    st.subheader("Locked 300-case replication")
    replication_metrics = st.columns(5)
    replication_metrics[0].metric(
        "Cases", str(replication["cohort"]["case_count"])
    )
    replication_metrics[1].metric(
        "Questions", str(replication["cohort"]["question_count"])
    )
    replication_metrics[2].metric(
        "Adaptive Top-1", f"{replication['retrieval']['adaptive']['hit@1']:.1%}"
    )
    replication_metrics[3].metric(
        "Verified Token-F1",
        f"{replication['semantic_evaluation']['verified_token_f1']:.3f}",
    )
    replication_metrics[4].metric(
        "Evidence support",
        f"{replication['semantic_evaluation']['evidence_support_rate']:.1%}",
    )
    st.caption(
        "The replication excludes every V1, V2, V2-confirmation, and V2.1 case. All "
        "retrieval, generation, and verifier settings were locked before these 900 questions."
    )

    st.subheader("V4.2 paired image-report confirmation")
    multimodal = json.loads(MULTIMODAL_V42_SUMMARY_PATH.read_text(encoding="utf-8"))
    multimodal_stats = json.loads(MULTIMODAL_V42_STATISTICS_PATH.read_text(encoding="utf-8"))
    multimodal_runtime = json.loads(MULTIMODAL_V42_RUNTIME_PATH.read_text(encoding="utf-8"))
    report_metrics = multimodal["metrics"]["report_only_bm25"]
    paired_metrics = multimodal["metrics"]["paired_biovil_t_shortlist_reranker"]
    mrr_stats = multimodal_stats["comparisons"]["mrr"]
    multimodal_columns = st.columns(5)
    multimodal_columns[0].metric("Confirmation cases", multimodal["split_case_count"])
    multimodal_columns[1].metric("Report MRR", f"{report_metrics['mrr']:.3f}")
    multimodal_columns[2].metric(
        "Paired MRR",
        f"{paired_metrics['mrr']:.3f}",
        delta=f"{mrr_stats['mean_difference']:+.3f}",
    )
    multimodal_columns[3].metric("Paired Hit@10", f"{paired_metrics['hit@10']:.1%}")
    multimodal_columns[4].metric(
        "Warm request",
        f"{multimodal_runtime['latency']['warm_paired_request_estimated_mean_ms']:.1f} ms",
    )
    st.success(
        "Primary MRR difference case-bootstrap 95% CI "
        f"[{mrr_stats['ci_low']:+.3f}, {mrr_stats['ci_high']:+.3f}]; "
        f"paired randomization p={mrr_stats['paired_randomization_p']:.4f}."
    )
    st.caption(
        "The fixed BioViL-T reranker uses image pixels only after BM25 creates a top-100 "
        "report shortlist. Hit@1 improved numerically but its interval crossed zero. This is "
        "paired evidence retrieval on IU-Xray, not autonomous image diagnosis."
    )

    st.subheader("Development-only prompt ablation")
    prompt_ablation = pd.read_csv(
        ROOT
        / "experiments"
        / "final_optimized"
        / "prompt_ablation"
        / "development_prompt_ablation.csv"
    )
    st.dataframe(
        prompt_ablation[
            [
                "prompt_mode",
                "draft_token_f1",
                "verified_token_f1",
                "evidence_support_rate",
                "abstention_rate",
                "nli_contradiction_count",
            ]
        ],
        width="stretch",
        hide_index=True,
    )

    with st.expander("RadGraph clinical entity results"):
        radgraph = pd.read_csv(
            ROOT
            / "experiments"
            / "final_optimized"
            / "radgraph"
            / "held_out_radgraph_summary.csv"
        )
        st.dataframe(radgraph, width="stretch", hide_index=True)

    st.markdown(
        '<div class="research-note">Research prototype only. Results measure report-grounded QA behavior and do not establish clinical diagnostic performance.</div>',
        unsafe_allow_html=True,
    )


st.title("Evidence-Checking Medical RAG")
st.caption(
    f"IU X-Ray / OpenI | {RUNTIME.full_case_count:,} available cases | "
    f"{RUNTIME.mode.title()} Mode"
)
if RUNTIME.is_demo:
    st.info(
        "Demo Mode uses three tracked software-demo cases, BM25 retrieval, deterministic "
        "extractive answers, and rule-based evidence checks. Frozen research results remain "
        "available in the Experiment results tab."
    )

live_tab, v6_tab, multimodal_tab, results_tab = st.tabs(
    ["Report workflows", "V6 confirmation demo", "Paired image demo", "Experiment results"]
)

with live_tab:
    with st.sidebar:
        st.header("Run settings")
        active_models = DEMO_MODEL_OPTIONS if RUNTIME.is_demo else MODEL_OPTIONS
        active_retrievers = {"BM25": "bm25"} if RUNTIME.is_demo else RETRIEVER_OPTIONS
        active_agents = (
            {"Lexical + negation rules": "rule", "Disabled": "off"}
            if RUNTIME.is_demo
            else AGENT_OPTIONS
        )
        workflow = st.segmented_control(
            "Workflow",
            options=["V2 patient-scoped", "V1 open-corpus stress test"],
            default="V2 patient-scoped",
        )
        threshold = 0.40
        route_policy = "locked_v2"
        if workflow == "V2 patient-scoped":
            route_policy = st.segmented_control(
                "Route policy",
                options=["Closed-loop V2.1", "Frozen V2 route"],
                default="Closed-loop V2.1",
            )
            model_label = st.selectbox("Generator", list(active_models))
            retriever_label = next(iter(active_retrievers))
            prompt_label = next(iter(PROMPT_OPTIONS))
            top_k = 6
            evidence_scope = "Patient-scoped evidence"
            agent_label = next(iter(active_agents))
            if route_policy == "Closed-loop V2.1":
                st.info(
                    "Post-submission V2.1: infer the report section, assess evidence, "
                    "retry once when needed, and abstain using a development-only threshold."
                )
            else:
                st.info(
                    "Frozen V2 policy: explicit case-ID scope, deterministic section route, "
                    "top-6 evidence, and advisory NLI audit."
                )
        else:
            retriever_label = st.selectbox("Retriever", list(active_retrievers))
            prompt_label = st.selectbox("Prompt", list(PROMPT_OPTIONS))
            model_label = st.selectbox("Generator", list(active_models))
            top_k = st.number_input("Top-K", min_value=1, max_value=10, value=5, step=1)
            evidence_scope = st.segmented_control(
                "Generation evidence",
                options=["Selected case", "All candidates (ablation)"],
                default="Selected case",
            )
            agent_label = st.selectbox("Evidence checker", list(active_agents))
            if active_agents[agent_label] == "rule":
                threshold = st.slider(
                    "Rule support threshold",
                    min_value=0.30,
                    max_value=0.80,
                    value=0.40,
                    step=0.05,
                )
        st.divider()
        st.caption("CUDA is used automatically when available.")

    selected_case_id = None
    selected_v2_type = None
    selected_v2_question = None
    if workflow == "V2 patient-scoped":
        cases, _ = load_bm25_resources()
        case_labels = {
            f"{case['case_id']} | {(case.get('indication') or 'No indication')[:80]}": str(
                case["case_id"]
            )
            for case in cases
        }
        default_label = next(
            (label for label, case_id in case_labels.items() if case_id == "CXR1004"),
            next(iter(case_labels)),
        )
        scope_col, task_col = st.columns([3, 2], gap="large")
        with scope_col:
            case_label = st.selectbox(
                "Patient case scope",
                list(case_labels),
                index=list(case_labels).index(default_label),
            )
            selected_case_id = case_labels[case_label]
        with task_col:
            task_label = st.selectbox("Planner task", list(V2_TASKS))
            selected_v2_type, selected_v2_question = V2_TASKS[task_label]

    examples = load_examples()
    example_by_label = {
        f"{item['qid']} · {item['question'][:90]}": item for item in examples[:60]
    }
    input_col, upload_col = st.columns([3, 2], gap="large")
    with input_col:
        if workflow == "V2 patient-scoped":
            initial_question = str(selected_v2_question)
        else:
            example_label = st.selectbox(
                "Example question", ["Custom question", *example_by_label]
            )
            initial_question = (
                ""
                if example_label == "Custom question"
                else example_by_label[example_label]["question"]
            )
        question = st.text_area(
            "Question",
            value=initial_question,
            height=120,
            placeholder="Enter a radiology report-grounded question",
            key=f"question_{workflow}_{selected_v2_type or 'v1'}",
        )
    with upload_col:
        uploaded = st.file_uploader("Question file", type=["txt", "json"])
        if workflow == "V2 patient-scoped":
            st.caption("JSON fields: case_id, question, and question_type. TXT overrides the question only.")
        else:
            st.caption("TXT: plain question. JSON: question and optional question_type.")

    run_clicked = st.button(
        "Run pipeline", type="primary", icon=":material/play_arrow:", width="content"
    )
    if run_clicked and workflow == "V2 patient-scoped":
        try:
            uploaded_question, uploaded_type, uploaded_case_id = parse_uploaded_scoped_request(
                uploaded
            )
            active_case_id = uploaded_case_id or str(selected_case_id)
            active_type = uploaded_type or str(selected_v2_type)
            closed_loop_enabled = route_policy == "Closed-loop V2.1"
            if not closed_loop_enabled and active_type not in {
                "case_scoped_findings",
                "case_scoped_impression",
                "case_scoped_summary",
            }:
                raise ValueError(f"Unsupported V2 question_type: {active_type}")
            active_question = uploaded_question or question.strip() or str(selected_v2_question)
            start = time.perf_counter()
            with st.status("Running patient-scoped pipeline", expanded=True) as status:
                locked_top_k, verifier_config = load_v2_configs()
                st.write(f"Applying explicit case-ID metadata filter: {active_case_id}")
                planning_trace: list[dict[str, Any]] = []
                answerability_threshold: float | None = None
                if closed_loop_enabled:
                    case_by_id, scoped_retriever = load_scoped_resources()
                    agent_result = ClosedLoopEvidenceAgent(scoped_retriever).run(
                        active_question, active_case_id
                    )
                    planning_trace = agent_result.trace
                    answerability_threshold = load_v21_answerability_threshold()
                    route_abstained = (
                        agent_result.answer_probability < answerability_threshold
                    )
                    retrieved = [
                        {
                            "rank": rank,
                            "case_id": active_case_id,
                            "chunk_id": chunk_id,
                            "section": section,
                            "position": rank,
                            "text": text,
                            "score": score,
                        }
                        for rank, (chunk_id, section, text, score) in enumerate(
                            zip(
                                agent_result.retrieved_chunk_ids,
                                agent_result.retrieved_sections,
                                agent_result.retrieved_texts,
                                agent_result.retrieved_scores,
                                strict=True,
                            ),
                            start=1,
                        )
                    ]
                    st.write(
                        f"Inferred route: {agent_result.final_intent}; "
                        f"answer probability: {agent_result.answer_probability:.3f}"
                    )
                    question_plan = {
                        "planned_intent": agent_result.planned_intent,
                        "final_intent": agent_result.final_intent,
                        "retried": agent_result.retried,
                    }
                else:
                    case_by_id, _ = load_scoped_resources()
                    route_abstained = False
                    st.write(f"Planner route: {expected_section(active_type)}")
                    retrieved = retrieve_scoped_evidence(
                        active_case_id, active_question, active_type, locked_top_k
                    )
                    question_plan = {
                        "question_type": active_type,
                        "section": expected_section(active_type),
                    }
                context = "\n".join(row["text"] for row in retrieved)
                prompt = build_scoped_live_prompt(active_case_id, active_question, retrieved)
                active_agent_mode = active_agents[agent_label]
                if route_abstained:
                    raw_answer = draft_answer = "NOT ANSWERABLE"
                    sentence_checks = []
                    support_rate = 0.0
                    st.write("Abstaining before generation because evidence is insufficient")
                else:
                    st.write("Generating evidence-only answer")
                    raw_answer, draft_answer = generate(prompt, active_models[model_label])
                    st.write("Auditing sentence-level grounding")
                if not route_abstained and active_agent_mode == "semantic":
                    checked = check_semantic_evidence_support(
                        draft_answer,
                        context,
                        load_semantic_predictor(),
                        min_combined_support=float(verifier_config["support_threshold"]),
                        entailment_threshold=float(verifier_config["entailment_threshold"]),
                        contradiction_threshold=float(verifier_config["contradiction_threshold"]),
                        lexical_weight=float(verifier_config["lexical_weight"]),
                    )
                    sentence_checks = [asdict(check) for check in checked.sentence_checks]
                    support_rate = checked.support_rate
                elif not route_abstained and active_agent_mode == "rule":
                    checked = check_evidence_support(
                        draft_answer,
                        context,
                        min_sentence_support=float(threshold),
                    )
                    sentence_checks = [asdict(check) for check in checked.sentence_checks]
                    support_rate = checked.support_rate
                elif not route_abstained:
                    sentence_checks = []
                    support_rate = 0.0
                status.update(label="Patient-scoped pipeline complete", state="complete", expanded=False)
            result = {
                "workflow": "v2_patient_scoped",
                "question": active_question,
                "question_plan": question_plan,
                "retriever": (
                    "closed_loop_agent_v2"
                    if closed_loop_enabled
                    else "case_scoped_agent_routed_bm25"
                ),
                "prompt_mode": "direct_evidence_only",
                "model": active_models[model_label],
                "top_k": locked_top_k,
                "evidence_scope": active_case_id,
                "threshold": (
                    answerability_threshold
                    if closed_loop_enabled
                    else float(verifier_config["support_threshold"])
                ),
                "agent_mode": active_agent_mode,
                "agent_action_policy": (
                    "closed_loop_answer_or_abstain"
                    if closed_loop_enabled
                    else ("audit_only" if active_agent_mode == "semantic" else active_agent_mode)
                ),
                "retrieval_decision": {
                    "source": "closed_loop_agent" if closed_loop_enabled else "patient_scope",
                    "abstained": route_abstained,
                    "reason": (
                        f"development threshold {answerability_threshold:.3f}"
                        if closed_loop_enabled and answerability_threshold is not None
                        else f"locked {expected_section(active_type)} route, top-{locked_top_k}"
                    ),
                },
                "raw_generation": raw_answer,
                "draft_answer": draft_answer,
                "final_answer": "NOT ANSWERABLE" if route_abstained else draft_answer,
                "support_rate": support_rate,
                "abstained": route_abstained,
                "sentence_checks": sentence_checks,
                "planning_trace": planning_trace,
                "retrieved_cases": retrieved,
                "scoped_case": case_by_id[active_case_id],
                "latency_seconds": time.perf_counter() - start,
                "prompt": prompt,
            }
            st.session_state["pipeline_result"] = result
        except Exception as exc:
            st.exception(exc)
        run_clicked = False
    if run_clicked:
        try:
            uploaded_question, uploaded_type = parse_uploaded_question(uploaded)
            active_question = uploaded_question or question.strip()
            if not active_question:
                st.warning("Enter a question or upload a question file.")
            else:
                start = time.perf_counter()
                with st.status("Running pipeline", expanded=True) as status:
                    st.write("Planning query")
                    plan = plan_question(active_question, uploaded_type)
                    st.write("Retrieving linked cases")
                    retrieved, selected_evidence, retrieval_decision = retrieve_with_policy(
                        plan.retrieval_query, active_retrievers[retriever_label], int(top_k)
                    )
                    generation_evidence = (
                        selected_evidence
                        if evidence_scope == "Selected case"
                        else retrieved
                    )
                    context = evidence_context(generation_evidence)
                    prompt = build_prompt(active_question, context, PROMPT_OPTIONS[prompt_label])
                    st.write("Generating draft answer")
                    raw_answer, draft_answer = generate(prompt, active_models[model_label])
                    st.write("Checking sentence-level evidence")
                    agent_mode = active_agents[agent_label]
                    if agent_mode == "semantic":
                        _, semantic_config = load_locked_configs()
                        checked = check_semantic_evidence_support(
                            draft_answer,
                            "" if not selected_evidence else evidence_context(selected_evidence),
                            load_semantic_predictor(),
                            min_combined_support=float(semantic_config["support_threshold"]),
                            entailment_threshold=float(
                                semantic_config["entailment_threshold"]
                            ),
                            contradiction_threshold=float(
                                semantic_config["contradiction_threshold"]
                            ),
                            lexical_weight=float(semantic_config["lexical_weight"]),
                        )
                        final_answer = checked.revised_answer
                        checks = [asdict(check) for check in checked.sentence_checks]
                        support_rate = checked.support_rate
                        abstained = checked.abstained
                    elif agent_mode == "rule":
                        checked = check_evidence_support(
                            draft_answer,
                            "" if not selected_evidence else evidence_context(selected_evidence),
                            min_sentence_support=float(threshold),
                        )
                        final_answer = checked.revised_answer
                        checks = [asdict(check) for check in checked.sentence_checks]
                        support_rate = checked.support_rate
                        abstained = checked.abstained
                    else:
                        final_answer = draft_answer
                        checks = []
                        support_rate = 0.0
                        abstained = False
                    status.update(label="Pipeline complete", state="complete", expanded=False)
                result = {
                    "question": active_question,
                    "question_plan": asdict(plan),
                    "retriever": active_retrievers[retriever_label],
                    "prompt_mode": PROMPT_OPTIONS[prompt_label],
                    "model": active_models[model_label],
                    "top_k": int(top_k),
                    "evidence_scope": evidence_scope,
                    "threshold": float(threshold),
                    "agent_mode": agent_mode,
                    "retrieval_decision": retrieval_decision,
                    "raw_generation": raw_answer,
                    "draft_answer": draft_answer,
                    "final_answer": final_answer,
                    "support_rate": support_rate,
                    "abstained": abstained,
                    "sentence_checks": checks,
                    "retrieved_cases": retrieved,
                    "latency_seconds": time.perf_counter() - start,
                    "prompt": prompt,
                }
                st.session_state["pipeline_result"] = result
        except Exception as exc:
            st.exception(exc)

    if "pipeline_result" in st.session_state:
        st.divider()
        render_pipeline_result(st.session_state["pipeline_result"])

with v6_tab:
    st.subheader("V6 confirmation: image-to-report demonstration")
    st.caption(
        "Upload a chest X-ray and retrieve the top-ranked candidate report from the frozen "
        "240-case V6 confirmation pool using the locked MedSigLIP max-chunk policy."
    )
    st.warning(
        "This is an interactive research demonstration. The retrieved report is a ranked "
        "candidate from a closed corpus; the system does not verify patient identity, diagnose "
        "from pixels, or establish clinical safety."
    )
    if RUNTIME.is_demo:
        st.info(
            "V6 confirmation demo needs the local OpenI source cases, official images, and "
            "MedSigLIP weights. It is disabled in lightweight Demo Mode."
        )
    else:
        v6_upload_col, v6_request_col = st.columns([2, 3], gap="large")
        with v6_upload_col:
            v6_image = st.file_uploader(
                "Chest X-ray image",
                type=["png", "jpg", "jpeg"],
                key="v6_image",
            )
            if v6_image is not None:
                st.image(v6_image.getvalue(), width=360)
                st.caption(v6_image.name)
        with v6_request_col:
            v6_indication = st.text_input(
                "Clinical indication",
                value="Chest pain",
                key="v6_indication",
            )
            v6_question = st.text_area(
                "Report-grounded question",
                value="What is the final radiology impression for this examination?",
                height=110,
                key="v6_question",
            )
            v6_answer_mode = st.selectbox(
                "Answer mode",
                [
                    "Extractive report answer (no generator download)",
                    "Qwen2.5-1.5B report-grounded answer",
                ],
                key="v6_answer_mode",
            )
            v6_run = st.button(
                "Run V6 candidate retrieval",
                type="primary",
                icon=":material/radiology:",
                key="run_v6_demo",
            )

        if v6_run:
            if v6_image is None:
                st.warning("Upload a chest X-ray image before running the V6 demonstration.")
            elif not v6_question.strip():
                st.warning("Enter a report-grounded question.")
            else:
                try:
                    started = time.perf_counter()
                    with st.status("Running V6 candidate-report workflow", expanded=True) as status:
                        st.write("Loading frozen V6 candidate pool and report chunks")
                        v6_resources = load_v6_dashboard_resources()
                        st.write("Encoding uploaded pixels with MedSigLIP-448")
                        v6_encoder = load_v6_image_encoder()
                        v6_embedding = encode_v6_uploaded_image(
                            v6_image.getvalue(), v6_encoder
                        )
                        st.write("Generating BM25 top-100 report shortlist")
                        v6_retrieved = retrieve_v6(
                            v6_indication,
                            v6_question,
                            v6_embedding,
                            v6_resources,
                            top_k=10,
                        )
                        v6_selected = v6_retrieved[0]
                        v6_evidence = "\n".join(
                            [
                                f"Case ID: {v6_selected['case_id']}",
                                f"Findings: {v6_selected['findings']}",
                                f"Impression: {v6_selected['impression']}",
                            ]
                        )
                        st.write("Preparing a report-grounded answer")
                        if v6_answer_mode.startswith("Extractive"):
                            v6_raw_answer = extractive_v6_answer(v6_question, v6_selected)
                            v6_answer = v6_raw_answer
                            v6_generation_mode = "extractive"
                        else:
                            v6_prompt = build_v6_generation_prompt(
                                v6_indication, v6_question, v6_selected
                            )
                            v6_raw_answer, v6_answer = generate(
                                v6_prompt,
                                MODEL_OPTIONS["Qwen2.5-1.5B (full experiment)"],
                            )
                            v6_generation_mode = "qwen2.5"
                        v6_audit = check_evidence_support(
                            v6_answer,
                            v6_evidence,
                            min_sentence_support=0.65,
                        )
                        status.update(
                            label="V6 candidate-report workflow complete",
                            state="complete",
                            expanded=False,
                        )
                    st.session_state["v6_demo_result"] = {
                        "workflow": "v6_confirmation_candidate_report_demo",
                        "question": v6_question.strip(),
                        "indication": v6_indication.strip(),
                        "image_name": v6_image.name,
                        "image_bytes": v6_image.getvalue(),
                        "candidate_count": len(v6_resources.candidate_ids),
                        "shortlist_size": int(
                            v6_resources.config["multimodal_retrieval"]["shortlist_size"]
                        ),
                        "text_weight": float(
                            v6_resources.config["multimodal_retrieval"]["text_weight"]
                        ),
                        "image_weight": float(
                            v6_resources.config["multimodal_retrieval"]["image_weight"]
                        ),
                        "encoder": "google/medsiglip-448",
                        "retrieved_cases": v6_retrieved,
                        "answer": v6_answer,
                        "raw_answer": v6_raw_answer,
                        "generation_mode": v6_generation_mode,
                        "support_rate": float(v6_audit.support_rate),
                        "abstained_by_audit": bool(v6_audit.abstained),
                        "audit": asdict(v6_audit),
                        "latency_seconds": time.perf_counter() - started,
                    }
                except Exception as exc:
                    st.exception(exc)

        if "v6_demo_result" in st.session_state:
            v6_result = st.session_state["v6_demo_result"]
            st.divider()
            v6_metrics = st.columns(6)
            v6_metrics[0].metric("Candidate pool", v6_result["candidate_count"])
            v6_metrics[1].metric("BM25 shortlist", v6_result["shortlist_size"])
            v6_metrics[2].metric("Top-ranked report", v6_result["retrieved_cases"][0]["case_id"])
            v6_metrics[3].metric("Evidence support", f"{v6_result['support_rate']:.1%}")
            v6_metrics[4].metric("Latency", f"{v6_result['latency_seconds'] * 1000:.0f} ms")
            v6_metrics[5].metric("Answer", v6_result["generation_mode"])

            v6_image_view, v6_report_view = st.columns([2, 3], gap="large")
            with v6_image_view:
                st.image(v6_result["image_bytes"], width=360)
                st.caption(v6_result["image_name"])
            with v6_report_view:
                st.markdown('<div class="evidence-label">Top-ranked candidate report</div>', unsafe_allow_html=True)
                v6_selected = v6_result["retrieved_cases"][0]
                st.write(f"Candidate case ID: {v6_selected['case_id']}")
                st.markdown('<div class="evidence-label">Findings</div>', unsafe_allow_html=True)
                st.write(v6_selected["findings"] or "Not reported")
                st.markdown('<div class="evidence-label">Impression</div>', unsafe_allow_html=True)
                st.write(v6_selected["impression"] or "Not reported")

            st.markdown('<div class="evidence-label">Report-grounded answer</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="answer-band">{html.escape(str(v6_result["answer"]))}</div>',
                unsafe_allow_html=True,
            )
            if v6_result["abstained_by_audit"]:
                st.info("The lexical audit marked the generated answer as insufficiently supported; this is an audit signal, not a clinical judgment.")

            st.subheader("V6 retrieval trace")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Rank": row["rank"],
                            "Case ID": row["case_id"],
                            "Fused": row["fused_score"],
                            "BM25": row["bm25_score"],
                            "BM25 normalized": row["bm25_normalized"],
                            "Image max-chunk": row["image_similarity"],
                            "Image normalized": row["image_normalized"],
                        }
                        for row in v6_result["retrieved_cases"]
                    ]
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "Fused": st.column_config.NumberColumn(format="%.3f"),
                    "BM25": st.column_config.NumberColumn(format="%.3f"),
                    "BM25 normalized": st.column_config.NumberColumn(format="%.3f"),
                    "Image max-chunk": st.column_config.NumberColumn(format="%.3f"),
                    "Image normalized": st.column_config.NumberColumn(format="%.3f"),
                },
            )
            v6_trace_rows = [
                {"Step": 1, "Action": "Encode uploaded image", "Output": "MedSigLIP-448 image embedding"},
                {"Step": 2, "Action": "Text shortlist", "Output": "BM25 over the frozen 240-case pool; top 100 retained"},
                {"Step": 3, "Action": "Multimodal reranking", "Output": "max chunk cosine; independent min-max; 0.5 text + 0.5 image"},
                {"Step": 4, "Action": "Evidence selection", "Output": "top-ranked candidate report only"},
                {"Step": 5, "Action": "Answer audit", "Output": "lexical report-support signal; not clinical adjudication"},
            ]
            st.dataframe(pd.DataFrame(v6_trace_rows), width="stretch", hide_index=True)
            v6_export = {key: value for key, value in v6_result.items() if key not in {"image_bytes", "audit"}}
            st.download_button(
                "Export V6 demo run",
                data=json.dumps(v6_export, indent=2, ensure_ascii=False),
                file_name="v6_candidate_report_demo_run.json",
                mime="application/json",
                icon=":material/download:",
            )
            st.markdown(
                '<div class="research-note">The V6 dashboard demonstrates closed-set candidate-report retrieval. It does not confirm patient identity, perform image diagnosis, or replace clinical review.</div>',
                unsafe_allow_html=True,
            )

with multimodal_tab:
    st.subheader("Paired chest X-ray and report retrieval")
    st.caption(
        "Upload a chest X-ray, retrieve a report with the locked V4.2 two-stage policy, "
        "then answer from the selected report and audit sentence-level support."
    )
    if RUNTIME.is_demo:
        st.info(
            "This tab needs the local official OpenI images, the 720-case BioViL-T index, "
            "and cached model weights. It remains disabled in lightweight Demo Mode."
        )
    else:
        upload_col, request_col = st.columns([2, 3], gap="large")
        with upload_col:
            multimodal_image = st.file_uploader(
                "Chest X-ray image",
                type=["png", "jpg", "jpeg"],
                key="multimodal_image",
            )
            if multimodal_image is not None:
                st.image(multimodal_image.getvalue(), width=360)
        with request_col:
            multimodal_indication = st.text_input(
                "Clinical indication",
                value="Chest pain",
                key="multimodal_indication",
            )
            multimodal_question = st.text_area(
                "Question",
                value="What is the final radiology impression for this examination?",
                height=110,
                key="multimodal_question",
            )
            multimodal_answer_mode = st.selectbox(
                "Answer mode",
                [
                    "Qwen2.5-1.5B + semantic evidence audit",
                    "Extractive answer + lexical evidence audit",
                ],
                key="multimodal_answer_mode",
            )
            multimodal_run = st.button(
                "Run paired retrieval",
                type="primary",
                icon=":material/radiology:",
                key="run_multimodal",
            )

        if multimodal_run:
            if multimodal_image is None:
                st.warning("Upload a chest X-ray image before running paired retrieval.")
            elif not multimodal_question.strip():
                st.warning("Enter a report-grounded question.")
            else:
                try:
                    started = time.perf_counter()
                    with st.status("Running paired image-report pipeline", expanded=True) as status:
                        st.write("Loading locked BioViL-T image-report index")
                        config, cases, candidate_ids, bm25, report_embeddings, encoder = (
                            load_multimodal_resources()
                        )
                        st.write("Encoding uploaded chest X-ray pixels")
                        image_embedding = encode_uploaded_image(multimodal_image.getvalue(), encoder)
                        st.write("Generating BM25 top-100 candidate set")
                        retrieved = paired_shortlist_retrieve(
                            question=multimodal_question.strip(),
                            indication=multimodal_indication.strip(),
                            candidate_ids=candidate_ids,
                            cases=cases,
                            bm25=bm25,
                            image_embedding=image_embedding,
                            report_embeddings=report_embeddings,
                            shortlist_size=int(config["reranking"]["shortlist_size"]),
                            text_weight=float(config["reranking"]["text_weight"]),
                            top_k=10,
                        )
                        st.write("Planning, generating, and checking report support")
                        selected_case = cases[retrieved[0]["case_id"]]
                        use_multimodal_qwen = multimodal_answer_mode.startswith("Qwen")
                        agent = answer_with_evidence_agent(
                            multimodal_question.strip(),
                            selected_case,
                            generator=generate if use_multimodal_qwen else None,
                            model_name=(
                                MODEL_OPTIONS["Qwen2.5-1.5B (full experiment)"]
                                if use_multimodal_qwen
                                else None
                            ),
                            semantic_checker=(
                                run_multimodal_semantic_check if use_multimodal_qwen else None
                            ),
                        )
                        status.update(
                            label="Paired image-report pipeline complete",
                            state="complete",
                            expanded=False,
                        )
                    st.session_state["multimodal_result"] = {
                        "workflow": "v4_2_paired_image_report",
                        "question": multimodal_question.strip(),
                        "indication": multimodal_indication.strip(),
                        "image_name": multimodal_image.name,
                        "image_bytes": multimodal_image.getvalue(),
                        "candidate_count": len(candidate_ids),
                        "shortlist_size": int(config["reranking"]["shortlist_size"]),
                        "text_weight": float(config["reranking"]["text_weight"]),
                        "answer_mode": multimodal_answer_mode,
                        "retrieved_cases": retrieved,
                        **agent,
                        "latency_seconds": time.perf_counter() - started,
                    }
                except Exception as exc:
                    st.exception(exc)

        if "multimodal_result" in st.session_state:
            result = st.session_state["multimodal_result"]
            st.divider()
            metrics = st.columns(6)
            metrics[0].metric("Candidates", result["candidate_count"])
            metrics[1].metric("Shortlist", result["shortlist_size"])
            metrics[2].metric("Text / image", "0.5 / 0.5")
            metrics[3].metric("Evidence support", f"{result['support_rate']:.1%}")
            metrics[4].metric("Latency", f"{result['latency_seconds'] * 1000:.0f} ms")
            metrics[5].metric("Answer mode", "Qwen" if result.get("generation_mode") == "qwen_non_oracle" else "Extractive")

            st.subheader("Grounded answer")
            answer_class = "research-note" if result["abstained"] else "answer-band"
            st.markdown(
                f'<div class="{answer_class}">{result["final_answer"]}</div>',
                unsafe_allow_html=True,
            )
            if result.get("generation_mode") == "qwen_non_oracle":
                st.caption("The answer was generated from the selected report and filtered by the locked semantic evidence checker.")

            image_view, evidence_view = st.columns([2, 3], gap="large")
            selected = result["retrieved_cases"][0]
            with image_view:
                st.image(result["image_bytes"], width=360)
                st.caption(result["image_name"])
            with evidence_view:
                st.markdown('<div class="evidence-label">Selected paired evidence</div>', unsafe_allow_html=True)
                st.write(f"Case {selected['case_id']}")
                st.markdown('<div class="evidence-label">Findings</div>', unsafe_allow_html=True)
                st.write(selected["findings"] or "Not reported")
                st.markdown('<div class="evidence-label">Impression</div>', unsafe_allow_html=True)
                st.write(selected["impression"] or "Not reported")

            st.subheader("Multimodal retrieval trace")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Rank": row["rank"],
                            "Case": row["case_id"],
                            "Fused": row["fused_score"],
                            "BM25": row["bm25_score"],
                            "Image similarity": row["image_similarity"],
                        }
                        for row in result["retrieved_cases"]
                    ]
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "Fused": st.column_config.NumberColumn(format="%.3f"),
                    "BM25": st.column_config.NumberColumn(format="%.3f"),
                    "Image similarity": st.column_config.NumberColumn(format="%.3f"),
                },
            )

            trace_rows = [
                {"Step": 1, "Action": "Encode image", "Output": "128-d BioViL-T embedding"},
                {"Step": 2, "Action": "Retrieve reports", "Output": "BM25 top 100"},
                {"Step": 3, "Action": "Rerank", "Output": "0.5 text + 0.5 image"},
                {
                    "Step": 4,
                    "Action": "Plan answer",
                    "Output": result["plan"]["answer_field"],
                },
                {
                    "Step": 5,
                    "Action": "Check evidence",
                    "Output": f"{result['support_rate']:.1%} supported",
                },
            ]
            st.dataframe(pd.DataFrame(trace_rows), width="stretch", hide_index=True)
            export = {key: value for key, value in result.items() if key != "image_bytes"}
            st.download_button(
                "Export paired run",
                data=json.dumps(export, indent=2, ensure_ascii=False),
                file_name="paired_image_report_run.json",
                mime="application/json",
                icon=":material/download:",
            )
            st.markdown(
                '<div class="research-note">Research retrieval prototype only. The image helps locate paired report evidence; the system is not an autonomous diagnostic device.</div>',
                unsafe_allow_html=True,
            )

with results_tab:
    render_results()
