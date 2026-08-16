# Final P2 Experiment Outputs

This folder contains precomputed P2 evidence-checking and P1-to-P2 ablation outputs.

Run the P2 threshold sweep:

```powershell
$env:PYTHONPATH = (Resolve-Path "src")
& ".\.venv\Scripts\python.exe" "scripts\run_evidence_threshold_sweep.py"
```

Run the P1 Stage 8B agent ablation:

```powershell
$env:PYTHONPATH = (Resolve-Path "src")
& ".\.venv\Scripts\python.exe" "scripts\run_p1_stage8b_agent_ablation.py"
```

The default P1 path is external to this repository and can be overridden with `--input`.

Build the 50-sentence manual threshold-calibration sample:

```powershell
& ".\.venv\Scripts\python.exe" "scripts\build_evidence_calibration_sample.py"
```

After filling `human_supported` in the generated CSV, compute manual checker metrics:

```powershell
& ".\.venv\Scripts\python.exe" "scripts\summarize_evidence_calibration.py"
```

See `docs/P2_MANUAL_CALIBRATION_GUIDE.md` for the annotation rule.
