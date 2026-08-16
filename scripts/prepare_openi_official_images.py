from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from medical_rag.multimodal.openi_images import resolve_official_image


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


def _download_range(url: str, destination: Path, start: int, end: int) -> Path:
    existing = destination.stat().st_size if destination.exists() else 0
    next_byte = start + existing
    expected_size = end - start + 1
    if existing > expected_size:
        raise RuntimeError(f"Range part is too large: {destination}")
    if existing == expected_size:
        return destination

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "wqf7023-medical-rag/0.2",
            "Range": f"bytes={next_byte}-{end}",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(f"Server ignored byte range {next_byte}-{end}: HTTP {response.status}")
        with destination.open("ab") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)

    if destination.stat().st_size != expected_size:
        raise RuntimeError(f"Incomplete byte range in {destination}")
    return destination


def download_with_resume(url: str, destination: Path, connections: int = 4) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = remote_size(url)
    existing = destination.stat().st_size if destination.exists() else 0
    if existing > expected:
        raise RuntimeError("Local archive is larger than the official remote file.")
    if existing == expected:
        return expected

    remaining = expected - existing
    connections = max(1, min(connections, remaining))
    chunk_size = (remaining + connections - 1) // connections
    ranges = []
    for index in range(connections):
        start = existing + index * chunk_size
        if start >= expected:
            break
        end = min(expected - 1, start + chunk_size - 1)
        part = destination.with_name(f"{destination.name}.part{index}")
        ranges.append((part, start, end))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = [executor.submit(_download_range, url, part, start, end) for part, start, end in ranges]
        for future in futures:
            future.result()

    assembling = destination.with_name(f"{destination.name}.assembling")
    with assembling.open("wb") as output:
        if existing:
            remaining_prefix = existing
            with destination.open("rb") as prefix:
                while remaining_prefix:
                    block = prefix.read(min(1024 * 1024, remaining_prefix))
                    if not block:
                        raise RuntimeError("Local prefix changed while the archive was assembled.")
                    output.write(block)
                    remaining_prefix -= len(block)
        for part, _, _ in ranges:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            part.unlink()

    if assembling.stat().st_size != expected:
        raise RuntimeError("Assembled archive length does not match the official file.")
    assembling.replace(destination)

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
            case_id = str(case["case_id"])
            for image in case.get("images", []):
                filename = image["filename"]
                declared_images += 1
                if resolve_official_image(case_id, filename, by_name) is not None:
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
    parser.add_argument("--connections", type=int, default=4)
    args = parser.parse_args()

    expected_size = remote_size(args.url)
    if not args.skip_download:
        download_with_resume(args.url, args.archive, connections=args.connections)
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
