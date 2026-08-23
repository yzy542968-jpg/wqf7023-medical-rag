# V9 Similar-Case Multimodal RAG: Related-Work and Technology Reuse Audit

## 1. Audit status

This audit was prepared after the V8 development no-go at repository commit
`1bee2d2`. It defines the permitted reuse boundary for a new V9 study. It does
not alter any V5-V8 configuration, cohort, result, or claim. No V9 confirmation
case IDs have been generated and no V9 outcome has been inspected.

V9 changes the task construct. The prior closed-set task asked whether the
paired report of a target image could be recovered from a controlled candidate
pool. V9 instead studies a clinically more plausible setting:

> Given a new chest radiograph, clinical indication, and medical question, can
> the system retrieve clinically similar image-report cases from other patients
> and use those cases to improve an evidence-grounded answer while the new
> patient's report remains hidden?

The historical image-report pair is the retrieval unit. The target report is
an offline reference only and is never an inference input.

## 2. Closest related work

| Work | What it contributes | Difference from V9 | Reuse decision |
|---|---|---|---|
| FactMM-RAG, NAACL 2025 | RadGraph/CheXbert factual-pair mining, a trained image-to-report retriever, and retrieval-augmented report generation | Generates reports from images; it is not clinical-indication-and-question-conditioned similar-case QA | Reuse factual-pair and evaluation concepts; adapt small MIT-licensed utilities only when they remove real duplication |
| CXR-RePaiR, ML4H 2021 | CLIP-style image-to-report retrieval, patient/study preprocessing, Top-k retrieval, and CheXbert evaluation | Retrieval-based report generation, no agentic QA | Reuse as the image-to-report retrieval baseline specification; port logic to the current stack rather than importing its old environment |
| X-REM, MIDL 2023 | Coarse retrieval followed by learned image-text matching and an NLI filter | ALBEF-based, multi-GPU, and report-generation focused | Reimplement the two-stage reranking idea with MedSigLIP; do not vendor the repository until every submodule license is audited |
| Concept-Enhanced Multimodal RAG (CEMRAG) | CXR-CLIP embeddings, FAISS, sparse medical concepts, and hierarchical evidence prompts | Report generation; cluster-oriented implementation | Reuse architecture ideas only because the repository has no detected top-level license |
| Multi-agent concept-bottleneck RAG, ECIR 2025 | Disease/concept prediction followed by specialist retrieval and report-writing agents | Uses hosted OpenAI components and disease documents; agents do not directly reason over the target pixels | Reuse only the separation of planning, retrieval, and synthesis; retain the local deterministic state machine |
| Multi-Agent Radiology VQA, 2025 preprint | Context-understanding, multimodal-reasoning, and answer-validation roles | No official code found in the targeted audit; not historical paired-case retrieval | Reimplement the role boundary, not code |
| Grounding CXR VQA with Generated Reports, 2025 preprint | A generated intermediate report improves downstream CXR VQA | Uses a report predicted from the same image, not other-patient cases | Reuse as a no-retrieval/intermediate-summary comparator if resources permit |
| Grounded Multimodal Case-Based Impression Drafting, 2026 preprint | Multimodal case similarity, FAISS retrieval, citation-constrained drafting, confidence refusal | Very close to V9, but drafts impressions rather than answering user questions; no official code found | Cite as the closest task-level precedent; adopt citation/refusal controls through an independent implementation |
| ReXVQA, 2025 | Large-scale chest-radiograph VQA taxonomy and evaluation cases | A benchmark rather than a similar-case RAG implementation | Reuse question categories and consider a licensed secondary evaluation adapter |

Primary sources:

- FactMM-RAG paper: <https://aclanthology.org/2025.naacl-long.28/>
- FactMM-RAG code: <https://github.com/cxcscmu/FactMM-RAG>
- CXR-RePaiR code: <https://github.com/rajpurkarlab/CXR-RePaiR>
- X-REM code: <https://github.com/rajpurkarlab/X-REM>
- CEMRAG code: <https://github.com/marcosal30/cemrag-rrg>
- Multi-agent CBM-RAG code: <https://github.com/tifat58/IRR-with-CBM-RAG>
- Multi-Agent Radiology VQA: <https://arxiv.org/abs/2508.02841>
- Generated-report-grounded CXR VQA: <https://arxiv.org/abs/2505.16624>
- Case-based impression drafting: <https://arxiv.org/abs/2603.17765>
- ReXVQA: <https://arxiv.org/abs/2506.04353>

## 3. Code-license boundary

FactMM-RAG and CXR-RePaiR are MIT-licensed. The ECIR multi-agent CBM-RAG
repository also contains an MIT license, but its hosted-API orchestration is a
poor technical fit. Any copied or substantially adapted MIT code must retain
the original copyright and permission notice and must be identified in a
repository third-party notice.

The targeted audit did not detect a top-level license for CEMRAG, X-REM, or
MMedAgent. Public visibility is not permission to reproduce or distribute
code. Their methods may be cited and independently reimplemented, but their
source files must not be copied into this repository without a completed
license audit.

Dataset and model licenses are separate from source-code licenses. In
particular, no MIMIC-derived text, image pixel, identifier, or artifact may be
committed to the public repository. Only manifests, hashes, aggregate metrics,
and synthetic test fixtures may be public when the applicable data agreement
permits them.

## 4. Existing repository capabilities to retain

| Capability | Existing implementation | V9 decision |
|---|---|---|
| BM25 retrieval | `src/medical_rag/retrieval/bm25_retriever.py` | Reuse as the text-only baseline |
| MedSigLIP image/text encoding | `src/medical_rag/multimodal/medsiglip.py` | Reuse frozen; add image-image and image-report case scores |
| Multi-view aggregation | `aggregate_view_embeddings` | Reuse unchanged |
| Long-report chunking | `src/medical_rag/multimodal/v6_chunking.py` | Reuse the frozen sentence-aware 64-token policy |
| Score normalization/fusion | `src/medical_rag/multimodal/fusion.py` | Extend to paired-case scoring while preserving deterministic tie handling |
| Retrieval metrics | `src/medical_rag/evaluation/metrics.py` | Extend with graded nDCG and candidate exclusion diagnostics |
| MedGemma generation | `src/medical_rag/multimodal/v6_generation.py` | Reuse model loading; define a new V9 evidence-constrained prompt |
| Evidence verification | Existing BioLinkBERT/verifier modules | Reuse as an automated signal, never as clinical gold |
| Agent control | `src/medical_rag/agentic/closed_loop_agent.py` and action policy | Extend with multimodal route selection and one bounded retry |
| Dashboard | Existing Streamlit app/runtime | Integrate only after the technical study is frozen |
| Grouped statistics | Existing V6/V7 statistical adapters | Reuse patient-grouped resampling when patient IDs are available |

## 5. New V9 components authorized

Only the following task-specific additions are authorized before a separate
protocol amendment:

1. A source-neutral paired-case record with patient ID, study ID, image paths,
   indication, findings, impression, labels, and RadGraph annotations.
2. A leakage auditor that verifies that the target study and all studies from
   the target patient are absent from the historical candidate bank.
3. Report-derived graded relevance based on active CheXbert-style labels and
   RadGraph facts, without rewarding agreement on absent labels.
4. Image-image, image-report, BM25, fixed multimodal, and small learned
   reranking baselines over the same historical bank.
5. Graded nDCG, Recall@K, MRR, label coverage, and target-patient leakage
   diagnostics.
6. An evidence-constrained prompt and output schema that distinguishes current
   image observations from historical analogies.
7. A bounded agentic controller with deterministic routing, one optional
   retrieval retry, verification, citation reporting, and abstention.
8. Synthetic fixtures and unit tests that exercise the full task contract
   without exposing restricted data.

## 6. Components explicitly rejected

- Copying an entire external repository into this project.
- Replacing the current stack with LangChain, LlamaIndex, CrewAI, or Qdrant.
- Sending protected radiology text or pixels to an online model or embedding
  API without a data agreement that explicitly permits it.
- Treating retrieved historical cases as proof that a finding exists in the
  target patient.
- Exposing the target findings, impression, report labels, or RadGraph facts to
  retrieval, prompting, routing, generation, or verification at inference.
- Calling OpenI case-ID separation patient-level independence.
- Using an unlicensed repository's code merely because it is public.
- Claiming first use of multimodal case-based RAG, citation-constrained
  drafting, or medical agents.

## 7. Research differentiation

V9's defensible contribution is the controlled combination and evaluation of:

1. a new-case query composed of chest radiograph, clinical indication, and
   medical question;
2. retrieval of other-patient historical image-report pairs;
3. question-conditioned answer generation rather than only full-report
   generation;
4. explicit comparison of no retrieval, text-only, image-only, fixed fusion,
   and learned fusion;
5. an aligned-versus-shuffled image negative control;
6. retrieval-to-answer transfer analysis; and
7. a bounded, auditable verification/retry/abstention controller.

The primary claim is not that RAG or agents are new. It is whether and under
what conditions multimodal similar-case retrieval improves reference-
consistent, evidence-grounded QA for a new patient case.

## 8. Audit outcome

The repository architecture is retained. FactMM-RAG and CXR-RePaiR provide
the strongest reusable methodological baselines. X-REM, CEMRAG, the recent
agentic VQA work, and the 2026 case-based drafting preprint inform independent
implementations and ablations. V9 must proceed protocol-first, and no final
confirmation cohort may be instantiated during development.
