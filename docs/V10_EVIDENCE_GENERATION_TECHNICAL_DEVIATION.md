# V10 Evidence Generation Technical Deviation

The initial prompt-only two-stage JSON execution was terminated after 144 of
2,256 planned Validation rows. Answer-stage strict JSON validity was 0/144,
support-stage strict JSON validity was 0/144, and every generation exhausted
its respective 64- or 96-token ceiling. Outputs contained natural-language or
reasoning text despite explicit JSON-only prompts.

The local failed rows are retained with SHA-256
`500c8813bf92fc88789db9fd9e49b52e47bef97232520255abaff0a176e5a232`.
No Calibration or Test data were read. The partial outputs are not used for
evidence-policy selection or scientific conclusions.

This is treated as a technical interface failure, not an unfavorable model
outcome. Continuing the same run would only repeat a deterministic formatting
failure. The revised execution policy was frozen separately before its
Validation outcomes were generated.

