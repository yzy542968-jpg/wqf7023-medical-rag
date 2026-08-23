# V9 CheXpert Plus Data Setup

## Purpose

CheXpert Plus is the preferred formal V9 source because it provides paired
chest radiographs and reports with study and patient identifiers. It also
provides CheXbert labels and RadGraph-XL annotations needed for the predefined
graded similar-case relevance measure.

The data are not redistributed in this repository. Obtain them from the
official Stanford source and accept the applicable data terms:

- Dataset page: <https://aimi.stanford.edu/datasets/chexpert-plus>
- Canonical download: <https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x?v=next>
- Official schema/annotation documentation:
  <https://github.com/Stanford-AIMI/chexpert-plus>

## Required Files

Place the following licensed files under the ignored local directory
`data/raw/chexpert_plus/`:

```text
data/raw/chexpert_plus/
|-- df_chexpert_plus_240401.csv
|-- train/
|   `-- patient.../study.../*.jpg
|-- valid/
|   `-- patient.../study.../*.jpg
|-- chexbert_labels/
|   `-- findings_fixed.json
`-- radgraph-XL-annotations/
    `-- section_findings.json
```

The image root must preserve the relative paths in the CSV `path_to_image`
column. Downloading a raster image release is sufficient for the planned local
MedSigLIP experiments; V9 does not require DICOM metadata as a model input.

## Source Audit

Run this command only after the official files are available:

```powershell
.\.venv\Scripts\python.exe scripts\audit_v9_chexpert_plus_source.py `
  --csv data\raw\chexpert_plus\df_chexpert_plus_240401.csv `
  --image-root data\raw\chexpert_plus `
  --chexbert-labels data\raw\chexpert_plus\chexbert_labels\findings_fixed.json `
  --radgraph-annotations data\raw\chexpert_plus\radgraph-XL-annotations\section_findings.json `
  --output data\splits\v9\v9_chexpert_plus_source_audit.json
```

The audit must show:

- non-zero patient and study counts;
- complete patient IDs;
- a non-zero `graded_qrels_eligible_study_count`;
- zero `graded_qrels_eligible_missing_image_count`;
- `graded_qrels_source_subset_ready = true`;
- `confirmation_ids_instantiated = false`.

Partial RadGraph findings coverage is expected. Formal V9 eligibility uses the
intersection of readable paired studies, CheXbert labels, and RadGraph facts.
This source audit does not create development, validation, or confirmation
case IDs.

## Data Handling Boundary

- Do not commit images, reports, patient/study manifests, or downloaded
  annotations.
- Keep all foundation-model inference local unless the data terms explicitly
  permit the selected service.
- Do not inspect or instantiate the final confirmation cohort during source
  setup.
- Do not use target findings, target impression, target labels, or target
  RadGraph facts as inference-time features.
