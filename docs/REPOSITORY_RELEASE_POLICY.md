# Repository Release Policy

## Included

- Source code, tests, configuration examples, and reproducibility scripts.
- Documentation, locked summary metrics, statistical outputs, manifests, and content fingerprints.
- Small processed QA benchmark manifests needed to audit exact splits.
- Final report and presentation deliverables with their validation boundaries stated explicitly.

## Excluded

- Python virtual environments and Hugging Face model caches.
- Raw OpenI/RadQA files and radiology image pixels.
- Dense embedding indexes and prompt-pack caches.
- Large intermediate generations, sentence-level model-score dumps, and browser QA screenshots.
- Tokens, passwords, private keys, Streamlit secrets, and machine-local environment files.

## Machine Audit

Build the release manifest from the repository root:

```powershell
& ".\.venv\Scripts\python.exe" scripts\build_submission_manifest.py
```

This command checks required files, locked-artifact SHA-256 values, tracked-file exclusions, GitHub's individual file-size limit, the resolved human-evaluation disposition, non-draft deliverables, the Git remote, environment versions, CUDA visibility, and the conditional V3 data state. A resolved disposition is either complete independent ratings or a declared `not_conducted` decision with zero ratings, an explicit limitation, and no human-score claim. Its output is stored at:

```text
experiments/final_submission/submission_manifest.json
```

For the final immutable release, run:

```powershell
& ".\.venv\Scripts\python.exe" scripts\build_submission_manifest.py --strict
git status --short
```

The strict audit must report `submission_ready: true`. `git status` must be clean after the final manifest is committed. No individual tracked file may reach GitHub's 100 MiB limit, and no restricted or secret material may be committed.

## Publication Gate

The repository may be published with human scoring marked `not_conducted` only when that limitation is visible and no human or clinical validation is claimed. The final thesis appendix must point to a working remote URL and a commit whose manifest hashes match the submitted DOCX, PDF, and PPTX. Official RadQA files must never be committed or redistributed.
