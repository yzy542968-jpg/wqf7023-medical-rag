"""Build readable contact sheets from rendered document page images."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def page_number(path: Path) -> int:
    match = re.search(r"page-(\d+)\.png$", path.name)
    if not match:
        raise ValueError(f"Unexpected page image name: {path.name}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--pages-per-sheet", type=int, default=12)
    args = parser.parse_args()

    pages = sorted(args.input_dir.glob("page-*.png"), key=page_number)
    if not pages:
        raise FileNotFoundError(f"No page images found in {args.input_dir}")

    columns = 3
    thumb_width = 420
    label_height = 32
    gutter = 18
    font = ImageFont.load_default(size=20)

    for start in range(0, len(pages), args.pages_per_sheet):
        batch = pages[start : start + args.pages_per_sheet]
        samples = [Image.open(path).convert("RGB") for path in batch]
        thumb_height = round(samples[0].height * thumb_width / samples[0].width)
        rows = (len(batch) + columns - 1) // columns
        sheet = Image.new(
            "RGB",
            (
                columns * thumb_width + (columns + 1) * gutter,
                rows * (thumb_height + label_height) + (rows + 1) * gutter,
            ),
            "#d8dde3",
        )
        draw = ImageDraw.Draw(sheet)

        for index, (path, page) in enumerate(zip(batch, samples, strict=True)):
            row, column = divmod(index, columns)
            x = gutter + column * (thumb_width + gutter)
            y = gutter + row * (thumb_height + label_height + gutter)
            page.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            sheet.paste(page, (x, y + label_height))
            draw.text((x, y + 4), f"Page {page_number(path)}", fill="#111827", font=font)

        first = page_number(batch[0])
        last = page_number(batch[-1])
        output = args.input_dir / f"contact-{first:02d}-{last:02d}.png"
        sheet.save(output, optimize=True)
        print(output)


if __name__ == "__main__":
    main()
