# Data Use and Licensing

The MIT license covers project-authored source code and documentation. It does not relicense OpenI/IU X-Ray, RadQA, model weights, or third-party mirrors.

## Repository Data Boundary

- Raw radiology files and image pixels are excluded from Git.
- The official `NLMCXR_png.tgz` archive and extracted OpenI PNG files remain local. Only its URL, size, checksum, counts, and filename-mapping manifest are versioned.
- `data/processed/sample_cases.jsonl` contains three software-demo records and is not research evidence.
- Versioned OpenI-derived benchmark manifests are de-identified research artifacts used to audit splits and frozen results.
- Credentialed RadQA files must remain under `data/raw/radqa/` and must never be committed or redistributed.
- Model weights and dense indexes are machine-local artifacts and are excluded from Git.

## Multimodal Model Boundary

V4.2 uses `microsoft/BiomedVLP-BioViL-T`. The Hugging Face model repository is MIT-licensed, while its model card limits intended use to vision-language research and reproducibility and places deployed diagnostic or medical-device use out of scope. The project downloads the official text and image weights locally, verifies the image-weight MD5 recorded by Microsoft, and does not redistribute either component.

Users are responsible for reviewing and complying with the original dataset and model terms before downloading or redistributing any external artifact. Cite the original OpenI/IU X-Ray publication for dataset use.
