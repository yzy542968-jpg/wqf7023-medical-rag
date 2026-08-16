# Dashboard Demonstration Script

## Recommended Settings

- Workflow: `V2 patient-scoped`
- Generator: `Qwen2.5-1.5B (full experiment)`
- Uploaded request: `data/demo/v2_patient_scoped_request.json`

The V2 policy is locked in the interface: explicit case-ID scope, deterministic section routing, top-6 evidence, and advisory Medical-NLI audit. The prototype has no authentication layer. No retrieval or verifier threshold should be changed during the demonstration.

## Three-Minute Flow

1. Open `Live pipeline` and identify the two workflows. Keep `V2 patient-scoped` selected.
2. Show that the user can select a patient case and Planner task manually.
3. Under `Question file`, upload `data/demo/v2_patient_scoped_request.json`.
4. Briefly show its three fields: `case_id`, `question_type`, and `question`.
5. Select `Run pipeline`.
6. Explain the status stages: patient scope, deterministic section route, locked top-6 retrieval, local Qwen generation, and advisory NLI audit.
7. Present the grounded answer and Agent trace. `Supported` means the automatic checker found adequate evidence; `Review` is a risk flag, not automatic deletion.
8. Show the retrieved evidence table. Every row must have the uploaded case ID and the routed report section.
9. Expand `Patient-scoped source report` to show the complete findings and impression.
10. Open `Experiment results` and show the V1 stress test separately from Benchmark V2.

## Claims to State

- The application runs on real IU X-Ray/OpenI reports.
- Patient identity is supplied as an explicit scope and is never guessed from clinical-text similarity in V2.
- The Planner route is deterministic: findings questions search findings, while impression and summary questions search impression.
- Calibration selected `top-k=6`; confirmation evidence recall is `99.4%`.
- Qwen2.5-1.5B reached Token-F1 `0.570` on the 120-case confirmation cohort, with case-bootstrap 95% CI `[0.556, 0.584]`.
- The extractive retrieved-context baseline reached `0.997`, so this controlled benchmark does not show a generation gain.
- Routed Hit@1 is a section-routing sanity check because routed candidates equal the relevance set; it is not semantic retrieval evidence.
- Automatic verifier rewriting reduced calibration F1, so the final V2 verifier is advisory and sends risk flags for review without silently altering the answer.
- The system is a report-grounded research prototype, not a chest X-ray diagnostic model.

## Claims to Avoid

- Do not compare V1 `0.206` and V2 `0.570` as if they were the same task.
- Do not describe deterministic section routing as autonomous or learned reasoning.
- Do not claim that routed Hit@1 proves retrieval quality or that Qwen improves over extraction in V2.
- Do not describe case-ID filtering as authentication or access control.
- Do not claim that Medical NLI scores are clinical correctness labels.
- Do not claim that the system analyzes the X-ray pixels; images remain linked metadata and previews.
- Do not claim state-of-the-art or deployment-ready clinical safety.
