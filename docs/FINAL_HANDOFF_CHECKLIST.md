# P2 Final Handoff Checklist

This checklist controls the immutable submission package. Do not change V1/V2 retrieval settings, prompts, thresholds, answer files, or headline claims.

## 1. Human-Evaluation Disposition

The blinded protocol was prepared but not conducted because no suitable independent reviewer was available before submission. The decision is frozen in `config/submission_decisions.json`.

Required conditions:

1. Both blinded 36-case packages remain at `0/36` completed rows.
2. No human score is inferred from automatic metrics.
3. The manuscript and deck state that no human or clinical validation is claimed.
4. The packages, keys, analysis script, and rating interface remain available only as a future-study protocol.

Rebuild the registry and confirm the policy status:

```powershell
& ".\.venv\Scripts\python.exe" scripts\build_final_results_registry.py
```

The registry must report `human_evaluation_policy.status: not_conducted` and zero completed rows for both packages.

## 2. Final Manuscript and Defence Deck

The final submission artifacts are:

```text
deliverables/22097191_ZHANG_YUE_P2_Research_Project.docx
deliverables/22097191_ZHANG_YUE_P2_Research_Project.pdf
deliverables/22097191_ZHANG_YUE_P2_Defence.pptx
```

The DOCX/PDF must contain five chapters, references, appendices, the frozen automated results, and the explicit human-evaluation limitation. The 15-slide deck must use the same claims and include a live-demo fallback. V3 remains future work unless all official RadQA splits are legally available and a complete baseline table is frozen.

## 3. GitHub Publication

The public repository is `https://github.com/yzy542968-jpg/wqf7023-medical-rag`. The verified author identity is configured and `main` is published. Do not publish raw data.

```powershell
git remote -v
git push origin main
git push origin p2-submission
```

The manuscript appendix must contain the working repository URL and the immutable `p2-submission` release reference before institutional upload.

## 4. Final Audit

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
& ".\.venv\Scripts\python.exe" -m compileall -q app.py human_evaluation_app.py scripts src
& ".\.venv\Scripts\python.exe" scripts\build_final_results_registry.py
& ".\.venv\Scripts\python.exe" scripts\build_submission_manifest.py --strict
```

The strict manifest must report `submission_ready: true`. Commit the final manifest and confirm a clean worktree. Rehearse the Dashboard at desktop resolution and retain screenshots or a short recording as fallback.

## 5. Institutional Submission

1. Confirm the supervisor's required signature or approval.
2. Complete the required Google Form and Turnitin submission.
3. Upload the final DOCX/PDF/PPTX and repository link to the required locations.
4. Keep an offline backup of the final artifacts, manifest, and Git commit identifier.
