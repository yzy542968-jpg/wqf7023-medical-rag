from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


DEFAULT_BASE_URL = (
    "https://huggingface.co/datasets/sasi2004/chest-xrays-indiana-university"
    "/resolve/main/images/images_normalized"
)


def _case_lookup(cases_path: Path) -> dict[str, dict[str, Any]]:
    return {case["case_id"]: case for case in load_cases_jsonl(cases_path)}


def _case_ids_from_retrieval(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [result["case_id"] for result in payload.get("results", [])]


def _validate_png(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image

        image = Image.open(path)
        image.verify()
        return {"valid": True, "size_bytes": path.stat().st_size}
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"valid": False, "error": str(exc), "size_bytes": path.stat().st_size if path.exists() else 0}


def _download(url: str, output_path: Path, retries: int = 3, timeout: int = 120) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    headers = {"User-Agent": "wqf7023-medical-rag/0.1"}

    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with temp_path.open("wb") as file:
                    while True:
                        chunk = response.read(1024 * 256)
                        if not chunk:
                            break
                        file.write(chunk)
            temp_path.replace(output_path)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            time.sleep(2 * attempt)


def build_manifest(
    cases_path: Path,
    output_dir: Path,
    case_ids: list[str],
    base_url: str = DEFAULT_BASE_URL,
    download: bool = True,
) -> dict[str, Any]:
    cases = _case_lookup(cases_path)
    manifest_cases: list[dict[str, Any]] = []

    for case_id in case_ids:
        case = cases.get(case_id)
        if not case:
            raise KeyError(f"Case id not found in cases file: {case_id}")

        images = []
        for image in case.get("images", []):
            filename = image["filename"]
            encoded = urllib.parse.quote(filename)
            url = f"{base_url}/{encoded}"
            local_path = output_dir / filename

            if download and not local_path.exists():
                _download(url, local_path)

            validation = _validate_png(local_path) if local_path.exists() else {"valid": False, "missing": True}
            images.append(
                {
                    "filename": filename,
                    "projection": image.get("projection", ""),
                    "url": url,
                    "local_path": str(local_path),
                    **validation,
                }
            )

        manifest_cases.append(
            {
                "case_id": case_id,
                "indication": case.get("indication", ""),
                "findings": case.get("findings", ""),
                "impression": case.get("impression", ""),
                "images": images,
            }
        )

    return {
        "source_dataset": "IU X-Ray / OpenI via Hugging Face mirror sasi2004/chest-xrays-indiana-university",
        "cases_path": str(cases_path),
        "image_output_dir": str(output_dir),
        "case_count": len(manifest_cases),
        "image_count": sum(len(case["images"]) for case in manifest_cases),
        "cases": manifest_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small OpenI image subset from Hugging Face.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--retrieval-json", type=Path)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--max-cases", default=5, type=int)
    parser.add_argument("--output-dir", default=Path("data/raw/images/images_normalized"), type=Path)
    parser.add_argument("--manifest", default=Path("data/processed/openi_image_subset_manifest.json"), type=Path)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    case_ids = list(args.case_id)
    if args.retrieval_json:
        case_ids.extend(_case_ids_from_retrieval(args.retrieval_json))

    unique_case_ids = []
    seen = set()
    for case_id in case_ids:
        if case_id not in seen:
            unique_case_ids.append(case_id)
            seen.add(case_id)

    unique_case_ids = unique_case_ids[: args.max_cases]
    if not unique_case_ids:
        raise ValueError("No case ids provided. Use --case-id or --retrieval-json.")

    manifest = build_manifest(
        cases_path=args.cases,
        output_dir=args.output_dir,
        case_ids=unique_case_ids,
        download=not args.no_download,
    )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote manifest for {manifest['case_count']} cases and {manifest['image_count']} images to {args.manifest}")


if __name__ == "__main__":
    main()

