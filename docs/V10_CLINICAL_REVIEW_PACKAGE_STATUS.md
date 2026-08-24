# V10 Independent Clinical Review Package Status

Status: package complete; independent clinical review pending.

The deterministic builder produced 100 unique Test cases and 400 blinded
answer presentations. Findings and impression each contribute 50 cases. The
four system identities are shuffled in a case-specific deterministic order and
stored only in a separate private key.

Local files:

- `experiments/v10_publication/v10_clinical_review_public.csv`: reviewer-facing
  questions, images, answers, evidence, and blank rating fields;
- `experiments/v10_publication/v10_clinical_review_private_key.csv`: private
  presentation-code to system mapping;
- `experiments/v10_publication/v10_clinical_reviewer_metadata.json`: blank
  reviewer qualification and completion metadata.

All six reviewer fields are blank in all 400 rows. The repository therefore
does not report clinical-review scores or imply that review occurred. A future
qualified reviewer must complete the public package without access to the
private key; metadata, exclusions, and missingness must then be completed
before unblinding and analysis.

## Artifact hashes

- Public package: `ada4e793d2b6185c4cdac9de4435139e4513b9e8da759a92cfe085c449f23c8d`
- Private key: `2be0e00345fe40b574095da967d98891275117e449d5b0db718e24d1a65d67dc`
- Reviewer metadata: `166e502cc88600ef30321a0439ecf626926643211384d4e098527b5d7f05a1f1`
- Frozen QA rows: `0e82b3cf5d3913fdac82f49b6742451cf095849cad88caa2c5bedb070f793944`
- Frozen retrieval rows: `68a1e5db7db21a0de258e30cb9f9c6f9cee892d89576c707b81ffb63a7617c91`

Package preparation improves readiness for external assessment but is not
itself evidence of clinical usefulness or safety.
