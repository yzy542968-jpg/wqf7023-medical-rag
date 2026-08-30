# Final-QA Confirmation Technical Deviations

## Pre-output lazy evidence-cache correction

The first confirmation invocation was interrupted before model loading and
before any Test output row was generated. Monitoring showed zero output rows,
approximately 13 MiB of GPU allocation and zero GPU utilization. The stack
trace showed that the shared runner was constructing P1 hierarchical evidence
for all 26,747 Test questions even though the frozen confirmation requested
only B3, B4 and B6.

The runner was corrected to construct evidence only for conditions explicitly
requested on the command line. No prompt, image, retrieval score, historical
report, generator, adapter, decoding parameter, output parser, policy,
hypothesis or endpoint changed. For B3, B4 and B6, evidence construction is
identical to the pre-correction code. The correction only avoids computing an
unused P1 cache.

This deviation occurred after the manifest was instantiated but before any
Test generation or outcome inspection. The corrected runner was subjected to
the full test suite and committed before confirmation generation resumed.

## Pre-output interpreter correction

The next invocation used the system Python 3.14 interpreter. It loaded the
frozen MedGemma checkpoint but stopped before adapter attachment because that
interpreter did not contain PEFT. No output row was generated. The study was
then resumed with the repository's existing `.venv` (Python 3.12), which
contains PEFT 0.20.0, Transformers 4.57.6, bitsandbytes 0.50.1 and CUDA-enabled
PyTorch. No package was installed or upgraded and no experimental parameter
changed.
