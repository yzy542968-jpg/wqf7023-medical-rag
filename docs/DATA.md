# Data Notes

The initial planned dataset is IU X-Ray / OpenI.

Expected raw files for the first implementation:

```text
data/raw/indiana_reports.csv
data/raw/indiana_projections.csv
data/raw/images/
```

The normalized project file will be:

```text
data/processed/openi_cases.jsonl
```

Each JSONL row represents one case-level retrieval unit:

```json
{
  "case_id": "CXR1",
  "indication": "...",
  "findings": "...",
  "impression": "...",
  "report_text": "...",
  "images": [
    {
      "filename": "CXR1_1_IM-0001-3001.png",
      "projection": "Frontal"
    }
  ]
}
```

If the downloaded dataset uses different column names, update `src/medical_rag/data/openi.py` rather than changing the project design.

## Current Real Dataset Status

As of 2026-05-23, the project has real IU X-Ray / OpenI metadata in place:

```text
data/raw/indiana_reports.csv
data/raw/indiana_projections.csv
data/processed/openi_cases.jsonl
```

Current counts:

- Reports / cases: 3,851
- Image projection rows: 7,466
- Case-level retrieval units generated: 3,851

The generated `openi_cases.jsonl` keeps each radiology report together with its associated image filenames. This is the core case-level retrieval unit for the project.

## Real Image Subset for Feasibility Demo

A small real-image subset has been downloaded for the first pneumonia retrieval demo:

```text
data/raw/images/images_normalized/
data/processed/openi_pneumonia_subset_manifest.json
experiments/openi_pneumonia_subset_contact_sheet.jpg
```

Current subset:

- Cases: 5
- Images: 10
- Query used to select cases: `right lower lobe pneumonia with cough and fever`

The image subset is intentionally small so P1 can show real paired image-report cases without requiring the full image download.

## Source Notes

The original dataset is the OpenI / Indiana University Chest X-ray collection. The current project copy uses a Hugging Face mirror for convenient metadata and PNG image access:

- OpenI: `https://openi.nlm.nih.gov/`
- Hugging Face mirror: `https://huggingface.co/datasets/sasi2004/chest-xrays-indiana-university`
- Kaggle mirror: `https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university`

For the proposal/report, cite the original OpenI / Indiana University dataset paper rather than only the mirror.
