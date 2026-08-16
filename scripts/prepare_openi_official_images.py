from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "wqf7023-medical-rag/0.2"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers["Content-Length"])


def download_with_resume(url: str, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = remote_size(url)
    existing = destination.stat().st_size if destination.exists() else 0
    if existing > expected:
        raise RuntimeError("Local archive is larger than the official remote file.")
    if existing == expected:
        return expected

    headers = {"User-Agent": "wqf7023-medical-rag/0.2"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing else "wb"
    with urllib.request.urlopen(request, timeout=120) as response, destination.open(mode) as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)

    actual = destination.stat().st_size
    if actual != expected:
        raise RuntimeError(f"Incomplete archive: expected {expected} bytes, found {actual}.")
    return actual


def inspect_archive(archive: Path) -> tuple[list[tarfile.TarInfo], int]:
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
    unsafe = [member.name for member in members if member.name.startswith(("/", "\\")) or ".." in Path(member.name).parts]
    if unsafe:
        raise RuntimeError(f"Unsafe archive paths detected: {unsafe[:3]}")
    png_count = sum(member.isfile() and member.name.lower().endswith(".png") for member in members)
    return members, png_count


def extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(destination, filter="data")


def build_pairing_manifest(cases_path: Path, image_root: Path) -> dict[str, object]:
    image_paths = list(image_root.rglob("*.png"))
    by_name: dict[str, Path] = {}
    duplicates = set()
    for path in image_paths:
        if path.name in by_name:
            duplicates.add(path.name)
        by_name[path.name] = path
    if duplicates:
        raise RuntimeError(f"Duplicate image basenames detected: {sorted(duplicates)[:3]}")

    case_count = 0
    declared_images = 0
    matched_images = 0
    missing_names = []
    with cases_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            case = json.loads(line)
            case_count += 1
            for image in case.get("images", []):
                filename = image["filename"]
                declared_images += 1
                if filename in by_name:
                    matched_images += 1
                else:
                    missing_names.append(filename)

    return {
        "case_count": case_count,
        "archive_png_count": len(image_paths),
        "declared_image_count": declared_images,
        "matched_image_count": matched_images,
        "missing_image_count": len(missing_names),
        "missing_image_examples": missing_names[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, verify, extract, and pair the official OpenI PNG archive.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--archive", type=Path, default=Path("data/raw/NLMCXR_png.tgz"))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw/openi_official_images"))
    parser.add_argument("--cases", type=Path, default=Path("data/processed/openi_cases.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/openi_multimodal_source_manifest.json"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    expected_size = remote_size(args.url)
    if not args.skip_download:
        download_with_resume(args.url, args.archive)
    if not args.archive.exists() or args.archive.stat().st_size != expected_size:
        raise RuntimeError("The local archive is absent or incomplete.")

    _, archive_png_count = inspect_archive(args.archive)
    if not args.skip_extract:
        extract_archive(args.archive, args.image_root)
    pairing = build_pairing_manifest(args.cases, args.image_root)
    if pairing["archive_png_count"] != archive_png_count:
        raise RuntimeError("Extracted PNG count does not match the archive.")

    manifest = {
        "source_dataset": "Indiana University Chest X-ray / OpenI",
        "official_url": args.url,
        "archive_size_bytes": expected_size,
        "archive_sha256": sha256_file(args.archive),
        "archive_png_count": archive_png_count,
        "pairing": pairing,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
