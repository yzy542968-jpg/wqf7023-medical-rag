# P2 Project Control

Updated: 2026-08-16
Submission deadline: 2026-08-20
Status: active, submission scope locked

## Final Objective

Deliver a reproducible and demonstrable master's research project that evaluates case-scoped, evidence-checking RAG for radiology report-grounded question answering without overstating autonomous-agent behavior or clinical safety.

The final contribution is the experimentally supported progression from open-corpus failure analysis to a controlled case-scoped workflow:

1. V1 quantifies patient-identification ambiguity, cross-case evidence contamination, retrieval headroom, and verifier limitations on real OpenI reports.
2. V2 demonstrates structural case isolation, deterministic report-section routing, locked evidence coverage, and calibration-supported advisory verification.
3. The validity audit explicitly shows that V2 is a workflow benchmark, not proof of semantic retrieval or generation superiority.

## Scope Lock

### Required for submission

- Real OpenI data provenance, processing, case-disjoint splits, and content fingerprints.
- V1 baselines, final system, confidence intervals, significance analysis, oracle analysis, contamination analysis, and verifier stress test.
- V2 controlled workflow, independent confirmation cohort, extractive baseline, validity audit, and advisory verifier policy.
- A demonstrable evidence-checking workflow with explicit `scope`, `retrieve`, `generate`, `audit`, and `review/abstain` stages.
- Frozen blinded human-evaluation files and an explicit, limitation-aware protocol disposition.
- Final P2 thesis/report, presentation, dashboard, reproducibility guide, and GitHub repository.

### Conditional enhancement

- RadQA V3 enters the final thesis only if credentialed data is legally available and a complete same-task baseline table can be finished without changing V1/V2 frozen results.
- Otherwise, RadQA is documented as the strongest follow-up study, not represented as completed work.

### Excluded from this submission

- Training or claiming a radiology image diagnostic model.
- Describing deterministic routing as learned or autonomous planning.
- Claiming authentication, access control, clinical validation, deployment readiness, or state of the art.
- Additional model sweeps that do not answer a predeclared research question.

## Delivery Schedule

| Date | Required output | Exit criterion |
|---|---|---|
| Aug 14 | Scope, validity, and artifact audit | Final claim hierarchy and blockers recorded |
| Aug 15 | Final experiment package | Frozen summaries, baselines, fingerprints, tests, and plots agree |
| Aug 16 | Human-evaluation decision | Protocol status recorded without fabricated or inferred scores |
| Aug 17 | Full P2 manuscript | Five chapters, references, appendices, limitations, and artifact links complete |
| Aug 18 | Presentation and dashboard | Claims match manuscript; full demo rehearsal passes |
| Aug 19 | Reproducibility and submission audit | Clean-environment instructions, GitHub, file naming, and PDF visual QA pass |
| Aug 20 | Submission | Final immutable submission bundle and backup verified |

## Acceptance Gates

### Scientific validity

- Every headline number maps to a locked artifact.
- Development, calibration, test, and confirmation roles are stated accurately.
- Extractive and no-retrieval baselines are included wherever the task permits them.
- Results distinguish statistical evidence, detector estimates, and unvalidated clinical interpretation.
- V1 and V2 are never presented as a paired same-task comparison.

### Agent claim

- The implementation exposes a trace of state transitions and tool decisions.
- Planner/routing rules are named deterministic when they are deterministic.
- Verifier output is a risk signal, not a clinical correctness label.
- Automatic answer rewriting remains disabled where calibration showed harm.

### Human-evaluation disposition

- The protocol is recorded as `not_conducted` because no suitable independent reviewer was available before submission.
- Both blinded packages remain unscored, and no score is inferred from automatic metrics.
- The manuscript and deck explicitly state that no human or clinical validation is claimed.
- The preserved protocol may be executed as future work without changing the frozen submitted results.

### Engineering and reproducibility

- Full tests and compilation pass.
- Dashboard passes desktop and mobile browser checks with no console errors or horizontal overflow.
- Raw restricted/large data, model caches, secrets, and virtual environments are excluded from Git.
- Configurations, seeds, environment versions, manifests, and output fingerprints are versioned.
- A fresh reader can reproduce preprocessing and evaluation from the documented commands.

### Submission artifacts

- Final Word/PDF manuscript has no placeholder text, broken tables, clipped figures, or inconsistent results.
- Slides fit the allocated defense time and contain a live-demo fallback.
- Appendix contains a working repository link and artifact map.
- Dashboard and manuscript use identical terminology and numbers.

## Risk Register

| Risk | Severity | Control |
|---|---|---|
| No independent human evaluation was conducted | High | Declare the limitation, report no human scores, and restrict claims to automated research evaluation |
| Remote GitHub repository is not published | High | Local Git content is audited; publish after Git identity and GitHub authentication are available |
| RadQA requires credentialed access | High | Treat as conditional; do not delay the submission-safe OpenI study |
| V2 task is structurally easy | High | Report extractive baseline and validity audit; position V2 as workflow control |
| NLI can false-flag composite grounded answers | High | Keep `audit_only`, show review status, and document calibration failure |
| Dashboard generation can be slow | Medium | Keep cached models, provide faster model option, screenshots, and recorded fallback |
| Numerical drift across report, slides, and app | Medium | Generate all final tables from locked JSON/CSV artifacts and run consistency audit |

## Current Status

- Automated V1/V2 experiments: complete and frozen.
- V2 structural-validity correction: complete.
- Test suite: 56 passing; compilation and dependency checks pass.
- Dashboard desktop/mobile regression: passed.
- Human evaluation: protocol preserved but not conducted; 0/36 rows completed in each package; no human score claimed.
- Final P2 manuscript: five chapters assembled; final DOCX and 23-page PDF produced and visually verified.
- P2 defence deck: final 15-slide PPTX completed from the P1 template; overflow and source-note audits pass, with all slides visually checked.
- RadQA V3: parser, patient-disjoint builder, evidence-sufficiency Agent, prompt-pack generation, answer metrics, runbook, and tests complete; official credentialed files remain unavailable.
- Git repository: published on `main` at `https://github.com/yzy542968-jpg/wqf7023-medical-rag`; release exclusions and tracked-content audit pass.
- Submission audit: machine-readable manifest builder accepts either completed ratings or the explicit non-conducted disposition. Final readiness still requires the remote repository and a final clean release state.

## Immediate Work Order

1. Run the strict submission audit and publish the immutable `p2-submission` tag.
2. Rehearse the live dashboard demonstration and retain screenshots or a recording as fallback.
3. Complete supervisor, Turnitin, Google Form, and institutional submission steps.

The exact handoff commands and final filenames are frozen in `docs/FINAL_HANDOFF_CHECKLIST.md`.
