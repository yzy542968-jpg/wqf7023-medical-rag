# Next-Stage Research Upgrade

Updated: 2026-08-14

## Decision

Keep V1 as the open-corpus failure analysis and V2 as the controlled case-isolation workflow. Do not use V2 routed Hit@1 or Qwen F1 as the main evidence for an advanced Agentic RAG contribution.

The next main benchmark should use independently authored clinical questions and independently annotated answer evidence. RadQA is the strongest text-first candidate: physicians created questions from referral information without seeing the answer context, it includes unanswerable questions, and it provides patient-level train/dev/test splits. Access requires PhysioNet credentialing and a data-use agreement.

Official sources:

- RadQA dataset: https://physionet.org/content/radqa/1.0.0/
- RadQA paper: https://aclanthology.org/2022.lrec-1.672/
- VQA-RAD paper and data description: https://www.nature.com/articles/sdata2018251

## Priority Experiments

1. **Natural-question benchmark:** reproduce RadQA baselines and retain answerable plus unanswerable items.
2. **Hard-negative retrieval:** mix the gold answer sentences with same-report distractors and clinically similar sentences from other permitted reports. Qrels must be independently annotated, not defined by the routed section.
3. **Planner evaluation:** infer intent and required evidence from free-form questions. Do not pass the gold question type to the planner.
4. **Agent actions:** implement `retrieve`, `rerank`, `answer`, `verify`, and `abstain` as explicit actions, then evaluate each action and its ablation.
5. **Comparable baselines:** no-context LLM, full-report LLM, extractive reader, BM25, dense retrieval, reranker, RAG without verifier, and complete agent on the same questions.
6. **Safety calibration:** add manually labelled unsupported, contradicted, and unanswerable cases; report risk-coverage and selective accuracy instead of treating NLI as clinical correctness.
7. **Human evaluation:** complete the existing frozen 36-case V2 sheet before making correctness or safety claims.

## Optional Multimodal Extension

VQA-RAD contains clinician-authored questions linked to radiology images and can support a true image-plus-text extension. It should be treated as a separate multimodal study because the current system only displays images and never supplies pixels to the model.

## Thesis Positioning

A defensible final thesis can present a progression:

1. V1 exposes patient-identification and cross-case contamination failures in open-corpus RAG.
2. V2 shows that explicit case scoping removes cross-case retrieval and that verifier rewriting must be calibrated conservatively.
3. V3 tests genuine question understanding, evidence retrieval, abstention, and agent actions on independently annotated natural questions.

This progression preserves the completed engineering work while moving the main scientific claim to a benchmark that can distinguish competing methods.
