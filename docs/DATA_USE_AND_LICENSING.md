# Data Use and Licensing

The MIT license covers project-authored source code and documentation. It does not relicense OpenI/IU X-Ray, RadQA, model weights, or third-party mirrors.

## Repository Data Boundary

- Raw radiology files and image pixels are excluded from Git.
- `data/processed/sample_cases.jsonl` contains three software-demo records and is not research evidence.
- Versioned OpenI-derived benchmark manifests are de-identified research artifacts used to audit splits and frozen results.
- Credentialed RadQA files must remain under `data/raw/radqa/` and must never be committed or redistributed.
- Model weights and dense indexes are machine-local artifacts and are excluded from Git.

Users are responsible for reviewing and complying with the original dataset and model terms before downloading or redistributing any external artifact. Cite the original OpenI/IU X-Ray publication for dataset use.
