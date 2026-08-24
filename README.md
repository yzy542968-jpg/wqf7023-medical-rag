# WQF7023 Medical RAG Project

**Research title:** Retrieval-Augmented Medical Question Answering over Paired Radiology Images and Reports

This repository contains the reproducible implementation and frozen evidence for Zhang Yue's WQF7023 Artificial Intelligence Research Project. The preserved V9 thesis study models a new chest-radiography case whose formal report is unavailable: the target image, pre-report indication, and question retrieve similar other-patient image-report pairs from a fixed historical bank, then a local multimodal generator answers with bounded historical support. The V10 publication extension adds duplicate-cluster-disjoint evaluation, fact-aware multiview reranking, fact-level evidence provenance, calibrated retrieval abstention, compact deterministic output assembly, and publication-oriented validity packages. It is not a diagnostic model, a clinically validated tool, or an authenticated clinical deployment.

**Repository:** https://github.com/yzy542968-jpg/wqf7023-medical-rag  
- **Historical submission release:** `p2-submission`
- **Preserved thesis release:** `v9-final-research-freeze`
- **Publication-extension release:** `v10-publication-freeze`
- **Current status:** V10 automated confirmation and software integration complete; independent clinical review and authorized external validation remain pending

## Study Structure

The final study deliberately separates two different tasks instead of treating their scores as directly comparable:

1. **V1 open-corpus stress test:** measures patient-identification ambiguity, cross-case evidence contamination, retrieval headroom, abstention, and verifier limitations on 120 real OpenI cases and 360 report-derived questions.
2. **V2 controlled case-scoped workflow:** evaluates deterministic section routing, patient isolation, evidence coverage, answer generation, and advisory verification on 720 previously unused OpenI cases, including a once-only 120-case confirmation cohort.
3. **V3 RadQA extension:** implements a natural-question, answerable/unanswerable benchmark and evidence-sufficiency Agent. Official results remain conditional because credentialed PhysioNet files are not present. Synthetic fixtures test software only and are never reported as research results.
4. **V4.2 paired image-report retrieval:** uses BM25 to retrieve 100 report candidates and BioViL-T image-report similarity to rerank them. The fixed policy is evaluated once on a disjoint 120-case confirmation cohort.
5. **V5 fresh-cohort multimodal QA:** adds indication ablations, a 100-permutation shuffled-image control, and non-oracle Qwen generation plus semantic evidence checking on a new 120-case confirmation cohort. V5 is a post-submission extension and does not modify the frozen P2 artifacts.
6. **V6 model-modernized confirmation:** repeats the alignment-specific retrieval test on a newly instantiated, broader within-source cohort with MedSigLIP, Qwen3-Embedding, Qwen2.5, and MedGemma 1.5. V6 keeps BM25 as the primary text baseline and the V5 verifier unchanged, so retrieval, generator, and verifier effects remain distinguishable.
7. **V7 adaptive-fusion extension:** trains a query-conditional text/image fusion model. Correct-image dependence replicated, but adaptive fusion did not exceed the validation-selected global weight; the mixed result is retained.
8. **V8 external-source development gate:** audits CheXpert Plus as an external replication source. The locally available 279-case validation subset did not meet the prespecified full-study requirement, so V8 stopped at a documented no-go.
9. **V9 new-patient other-patient similar-case RAG:** reallocates the complete 3,851-case OpenI source, trains an 865-parameter multimodal reranker on a 2,608-case historical bank, confirms retrieval on 752 Test cases, and evaluates target-image QA on 685 complete-reference Test cases. The target report is hidden at inference and is never retrieved as evidence.
10. **V10 publication extension:** clusters exact and near-duplicate reports before splitting, trains only on cluster-disjoint Train cases, evaluates a fact-aware multiview R5 ensemble on untouched Test cases, calibrates a no-reliable-history option, and separates target-image answers from deterministic case/section/fact provenance.

The demonstrated Agent follows explicit `scope`, `retrieve`, `generate`, `audit`, and `review/abstain` states. Routing rules are deterministic, and the verifier is a risk signal rather than a clinical correctness label.

## Frozen Results

### V1 Open-Corpus Stress Test

- Real OpenI cases/questions: 120 / 360; grouped development/test split: 84 / 36 cases.
- Held-out final verified Token-F1: `0.206` with case-bootstrap 95% CI `[0.167, 0.246]`.
- Final versus Case-BM25: `+0.035`; unadjusted paired randomization `p=0.0145`, Holm-adjusted `p=0.0870` across exploratory comparisons.
- Held-out hybrid retrieval: Hit@1 `0.287`, Hit@20 `0.509`, MRR `0.331`.
- Oracle target-case retrieval raises verified Token-F1 to `0.425`, identifying retrieval as the main bottleneck.
- The automated contamination detector estimates cross-case support in `19.4%-28.9%` of sentences and `57.4%-65.7%` of answers. These are detector estimates, not human-confirmed labels.

### V2 Controlled Workflow

- Main/confirmation cases: 600 / 120, all disjoint from V1.
- Development-selected `top-k=6`; confirmation evidence recall `0.994`.
- Confirmation Qwen Token-F1: `0.570`, 95% CI `[0.556, 0.584]`.
- Extractive retrieved-context baseline Token-F1: `0.997`; Qwen minus extractive: `-0.427`.
- Routed candidate pools equal qrels by construction, so routed Hit@1 is a routing sanity check rather than semantic retrieval evidence.
- Automatic verifier rewriting reduced calibration Token-F1; the frozen action is therefore `audit_only`.

All headline values are generated from locked artifacts in `experiments/final_submission/final_results_registry.json`.

### V4.2 Paired Image-Report Extension

- Official NLM image archive: 7,470 PNG files; all 7,466 normalized image references matched.
- Fixed candidate pool: 720 cases; development/confirmation: 600 / 120 cases.
- Confirmation report-only BM25 MRR: `0.556`; paired shortlist-reranker MRR: `0.596`.
- Paired MRR difference: `+0.040`, case-bootstrap 95% CI `[+0.014, +0.068]`.
- Confirmation Hit@10: `0.664 -> 0.736`; Token-F1: `0.594 -> 0.645`.
- Hit@1 improved numerically from `0.497` to `0.522`, but its 95% interval crosses zero.
- Warm paired request estimate on the local RTX 5070 Laptop GPU: `16.9 ms`; loaded model memory: approximately `526 MiB`.

V4.2 demonstrates that pixels can improve paired evidence retrieval when used for constrained reranking. Image-only retrieval remains weak, and the experiment does not establish diagnostic performance on new patients.

### V5 Fresh-Cohort End-to-End QA Extension

- Fresh OpenI cohort: 240 cases, split into 120 development and 120 confirmation cases; 360 confirmation questions.
- Indication + question BM25 confirmation MRR: `0.6590`; correct-image reranking MRR: `0.6971`.
- Correct-image minus BM25 MRR difference: `+0.0381`, case-bootstrap 95% CI `[+0.0159, +0.0614]`.
- Across 100 shuffled-image permutations, mean MRR was `0.5659`; no shuffled permutation reached the correct-image MRR (plus-one Monte Carlo `p=0.0099`).
- Non-oracle Qwen verified Token-F1 improved from `0.3563` to `0.3865`, difference `+0.0302`, case-bootstrap 95% CI `[+0.0101, +0.0511]`.
- V5 remains a closed-set paired-report retrieval and report-grounded QA study, not new-patient diagnosis or clinical validation.

### V6 Model-Modernized Confirmation

- Confirmation candidate pool: 240 case IDs, with 120 targets and 120 distractors; targets produced 360 deterministic report-derived questions.
- The pool contained 172 report-indexed normal and 68 report-indexed abnormal cases. These labels come from the dataset `problems` field and are not new clinical adjudications.
- BM25 indication-plus-question MRR: `0.6168`; MedSigLIP reranking MRR: `0.6474`; difference `+0.03069`, case-bootstrap 95% CI `[+0.00902, +0.05368]`.
- Correctly aligned MedSigLIP exceeded all 100 deterministic shuffled-image controls; plus-one Monte Carlo `p=0.00990`.
- Verified Token-F1 improved under both Qwen2.5 (`+0.01206`) and MedGemma 1.5 (`+0.03857`) when the same generators received the MedSigLIP-selected report.
- V6 is complete evidence for a within-source, closed-set paired-report confirmation. It is not external validation, patient-level independence verification, image diagnosis, clinical utility, or deployment safety evidence.

### V7 Adaptive-Fusion Result

- Global `alpha*=0.52` MRR: `0.6134`; adaptive query-conditional MRR: `0.6019`.
- Adaptive minus global: `-0.0115`, 95% case-bootstrap CI `[-0.0268,+0.0031]`; adaptive superiority did not pass.
- Correctly aligned adaptive retrieval still exceeded the shuffled-image distribution (`p=0.0198`).
- This mixed result motivated V9's graded-similarity task and candidate-level reranker rather than post-confirmation V7 tuning.

### V9 New-Patient Similar-Case RAG

- Full OpenI source: 3,851 cases; primary split: 2,631 Train / 376 Validation / 752 Test; historical bank: 2,608 report-bearing Train cases.
- Retrieval nDCG@10: BM25 `0.1342`, image-image `0.3156`, fixed multimodal `0.2469`, learned MLP `0.3279`.
- Learned minus image-only nDCG@10: `+0.01238`, 95% case-bootstrap CI `[+0.00923,+0.01558]`.
- Correctly aligned learned retrieval exceeded all 100 shuffled-image controls (plus-one `p=0.00990`).
- QA frame: 685 Test cases, 1,370 findings/impression questions, 5,480 local MedGemma generations.
- Token-F1: no retrieval `0.1456`, BM25 RAG `0.1479`, fixed multimodal RAG `0.1791`, learned multimodal RAG `0.1848`.
- Learned multimodal RAG minus no retrieval: `+0.03924`, 95% case-bootstrap CI `[+0.03257,+0.04574]`.
- Learned minus fixed multimodal QA was only numerical: `+0.00571`, CI `[-0.00096,+0.01228]`.
- The bounded agent reduced automated unsupported historical-support rows from `16.42%` to `0%` through one backup route or removal of the historical-support field; it did not verify target-image diagnoses.

V9 is the final primary technical study. It models a new patient whose report is unavailable, retrieves other-patient analogies, and separates target-image answers from historical support. It does not establish physician-adjudicated similarity, clinical diagnostic accuracy, safety, external generalization, or deployment utility.

### V9 Supplemental Validity and Robustness Audits

These analyses were committed under a post-hoc protocol and did not alter any frozen V9 model, prompt, parameter, output, or primary result.

- Cross-split similarity audit: 187 Test cases had Train-report cosine similarity at least `0.95`. After excluding them, the learned reranker remained first at nDCG@10 `0.2797` versus image-image retrieval `0.2646`.
- Qrel construct sensitivity: the learned reranker ranked first under label-only, RadGraph-fact-only, and frozen combined relevance definitions.
- Modern text baseline: Qwen3-Embedding-0.6B reached nDCG@10 `0.1956`, above BM25 `0.1342` but below learned multimodal retrieval `0.3279`.
- Wording robustness: canonical-to-paraphrase Top-1 consistency was `0.114` for BM25, `0.357` for Qwen3-Embedding, and `0.997` for the learned multimodal reranker.
- Clinical semantic overlap: F1-RadGraph showed learned multimodal RAG above BM25 RAG by `+0.02094`, 95% case-bootstrap CI `[+0.01286,+0.02899]`; differences from no retrieval and fixed multimodal RAG were unresolved.
- Structured-output audit: robust reparsing recovered `0` additional outputs; `2,943/5,480` remained incomplete, identifying output truncation as a genuine engineering limitation rather than a regex artifact.
- Researcher review: all 24 assistant-proposed exploratory labels were accepted without modification; none were excluded. This was researcher review, not independent radiologist adjudication.

The consolidated interpretation is in `docs/V9_SUPPLEMENTAL_VALIDITY_RESULTS.md`. The final manuscript retains these analyses as validity qualifications rather than new confirmatory hypotheses.

### V10 Cluster-Disjoint Publication Extension

- The 3,851-case source was partitioned by 3,013 exact/near-duplicate clusters: 2,510 Train, 383 Calibration, 384 Validation, and 574 Test cases, with zero cluster overlap between partitions.
- Six frozen Test cases had unusable empty-report RadGraph records and were retained as documented technical exclusions without replacement; 568 cases remained technically eligible.
- Retrieval nDCG@10: R4 nine-feature reranker `0.34905`; R5 fact-aware multiview ensemble `0.36007`; difference `+0.01103`, case-bootstrap 95% CI `[+0.00770,+0.01441]`.
- Correctly aligned R5 retrieval exceeded all 100 deterministic fixed-point-free shuffled-image assignments: aligned `0.36007`, shuffled mean `0.24963`, plus-one Monte Carlo `p=0.00990`.
- Retrieval confidence on Test achieved Brier `0.16739`, ECE `0.04579`, and AUROC `0.70546`; the frozen 80%-coverage threshold yielded observed coverage `81.51%`.
- Token-F1: target image without history `0.14942`; R4 whole-report RAG `0.20752`; R5 hierarchical RAG `0.20919`; calibrated selective RAG `0.20498`.
- R5 hierarchical RAG minus no-history generation was `+0.05978`, 95% CI `[+0.05114,+0.06860]`. Its advantage over R4 whole-report RAG was only `+0.00167`, CI `[-0.00347,+0.00683]`.
- Complete F1RadGraph improved from `0.08265` without history to `0.11053` with R5 hierarchical RAG. The R5-minus-R4 complete-score interval was marginally above zero, while entity and entity-relation intervals crossed zero.
- Deterministic assembly achieved `100%` schema and citation validity, but raw answer token-ceiling rates remained high. This is disclosed as a generator limitation rather than hidden by reparsing.
- A blinded 100-case, 400-row clinical-review package is prepared with empty rating fields and a separate private key. Its status is `pending_independent_review`; no reviewer result is claimed or fabricated.
- The MIMIC-CXR adapter and patient-level external-validation runbook are implemented, but no authorized external dataset was present for this release. No external result is claimed.

V10 strengthens internal validity and traceability. Its automated metrics measure retrieval relevance and report-reference consistency, not physician-adjudicated diagnostic correctness, safety, or clinical utility.

## Submission Status

Automated V1-V10 experiments, supplemental validity audits, and Dashboard integration are complete in the corresponding repository history. The V9 researcher accepted all 24 assistant-proposed exploratory labels without modification; this was not independent clinical adjudication. V10 therefore prepares a separate blinded clinical package and leaves every rating blank pending a qualified independent reviewer.

The final V9 reporting artifacts are:

```text
deliverables/22097191_ZHANG_YUE_V9_Final_Research_Project.docx
deliverables/22097191_ZHANG_YUE_V9_Final_Research_Project.pdf
deliverables/22097191_ZHANG_YUE_V9_Final_Defence.pptx
```

The earlier `P2_Research_Project` and `P2_Defence` files remain preserved as
historical pre-V9 artifacts. They should not be used as the current V9 result
summary.

The decision is recorded in `config/submission_decisions.json` and propagated into the result registry, manuscript, deck, and release audit. The blinded packages and rating interface are preserved for a future extension; they are not evidence for the submitted study.

## Repository Layout

```text
config/           Versioned experiment configuration examples
data/raw/         Original datasets; excluded from Git
data/processed/   Small normalized benchmarks and split manifests
data/sample/      Public software-only fixtures
deliverables/     Rendered report and presentation artifacts
docs/             Methods, results, runbooks, and submission controls
experiments/      Frozen metrics, audits, and evaluation packages
scripts/          Reproducible command-line entry points
src/medical_rag/  Data, retrieval, Agent, and evaluation modules
tests/            Automated regression and validity tests
```

## Reproduce the Audited Package

From PowerShell in the repository root:

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" -m pytest -q
& ".\.venv\Scripts\python.exe" -m compileall -q app.py human_evaluation_app.py scripts src
& ".\.venv\Scripts\python.exe" scripts\build_submission_manifest.py
```

Reproduce the lightweight V9 supplemental audits from the frozen local artifacts with:

```powershell
& ".\.venv\Scripts\python.exe" scripts\audit_v9_cross_split_duplicates.py
& ".\.venv\Scripts\python.exe" scripts\audit_v9_qrel_sensitivity.py
& ".\.venv\Scripts\python.exe" scripts\audit_v9_structured_output_reparse.py
& ".\.venv\Scripts\python.exe" scripts\build_v9_final_manuscript.py
& ".\.venv\Scripts\python.exe" scripts\build_v9_final_docx.py
```

The F1-RadGraph and Qwen3-Embedding audits additionally require their pinned local model snapshots. Their committed aggregate summaries remain inspectable without downloading weights.

Reproduce the V10 publication-extension software and lightweight checks with:

```powershell
& ".\.venv\Scripts\python.exe" scripts\build_v10_cluster_disjoint_split.py --help
& ".\.venv\Scripts\python.exe" scripts\run_v10_confirmation_retrieval.py --help
& ".\.venv\Scripts\python.exe" scripts\run_v10_confirmation_qa.py --help
& ".\.venv\Scripts\python.exe" scripts\evaluate_v10_confirmation_radgraph.py --help
& ".\.venv\Scripts\python.exe" -m pytest -q --basetemp outputs/pytest-v10
```

The audited final suite contains **252 passing tests**. Model-backed confirmation commands require the pinned local snapshots and non-public large artifacts documented by the V10 protocols; aggregate summaries remain versioned for inspection.

The manifest command verifies required files, locked SHA-256 values, tracked-file exclusions, file-size limits, the declared human-evaluation disposition, repository publication, and conditional V3 status. Use `--strict` only for the final submission gate; it exits unsuccessfully while external requirements such as the remote repository remain pending.

Raw OpenI files are not redistributed. Data acquisition, expected filenames, and processing commands are documented in `docs/DATA.md`. Exact selected configurations, seeds, split fingerprints, and locked outputs are versioned in `config/`, `data/splits/`, and `experiments/`.

## Interactive Dashboard

The repository now starts in **Demo Mode** after a fresh clone. Demo Mode uses three tracked software-only cases, BM25 retrieval, deterministic extractive answers, and rule-based evidence checking; it does not require raw data, model weights, or a GPU. When the full local OpenI case file is present, the application automatically enables Full Mode. Set `MEDICAL_RAG_DEMO_MODE=1` to force the lightweight path.

Launch the research dashboard:

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
```

Check runtime data readiness with `python scripts/preflight_runtime.py`; add `--require-full` when the full OpenI case file and dense index are mandatory.

Launch the system-blinded rating interface separately:

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run human_evaluation_app.py --server.port 8502
```

The main Dashboard exposes report workflows, frozen result tables, the earlier paired demos, and both V9 and V10 workflows. In V10 Full Mode, one or two uploaded chest X-ray views, an indication, and a question retrieve Top-3 reports from the 2,506-case cluster-disjoint Train bank with the frozen R5 ensemble. The interface displays calibrated retrieval confidence, abstains from claiming reliable history below the frozen threshold, and preserves case/section/fact provenance. The optional local MedGemma path produces a bounded target-image answer before deterministic support assembly. The rating application does not load the system-identity keys.

For an editable install, use `python -m pip install -e ".[all]"`. Exact direct versions from the audited machine are recorded in `requirements-lock.txt`. GitHub Actions runs compilation, all unit tests, and the fresh-clone Dashboard smoke test without downloading model weights.

## V3 RadQA

Official RadQA files, when legally obtained, belong at `data/raw/radqa/train.json`, `dev.json`, and `test.json`. Run `docs/BENCHMARK_V3_RADQA_RUNBOOK.md` exactly as written. Until the complete official baseline table is produced, V3 is reported only as implemented future validation, not as a completed experiment.

## Release Boundary

Do not commit raw radiology files, image pixels, model weights, caches, generated prompt packs, secrets, or virtual environments. The MIT license covers project-authored code, not third-party datasets or model weights. See `docs/DATA_USE_AND_LICENSING.md`, `docs/REPOSITORY_RELEASE_POLICY.md`, and the generated `experiments/final_submission/submission_manifest.json` before publishing.

Post-submission improvements and explicitly deferred independent clinical evaluation are documented in `docs/POST_SUBMISSION_RESEARCH_ROADMAP.md`. V5-V8 remain frozen historical studies. The V9 thesis result is frozen in `docs/V9_TECHNICAL_FREEZE.md`; the completed researcher review and post-hoc validity audits are separately versioned and did not modify that freeze. V10 is a separate publication extension governed by its own development, confirmation, deviation, metrics, review-status, external-validation, and technical-freeze records.

Methods and results for the v2.1 hard benchmark, two reserved wording-transfer tests, frozen v2.2 semantic planner, preregistered v2.3 hybrid planner, and 300-case locked replication are in `docs/POST_SUBMISSION_EXPERIMENTS.md`. V2.3 preserves the original result and improves transfer Macro F1, but raises false-answer risk on the second wording set; it is therefore reported as a robustness/safety trade-off rather than promoted as an unqualified replacement.
