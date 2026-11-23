"""Resize H&M product images to 256x256 (white padding, no distortion) and cache them.

Run from the project root:  python src/data/make_images.py
"""
import zipfile
from pathlib import Path

import polars as pl          # NEW
from PIL import Image
from tqdm import tqdm        # NEW

ROOT = Path(__file__).resolve().parents[2]
RAW_ZIP = ROOT / "data" / "raw" / "h-and-m-personalized-fashion-recommendations.zip"
LABELS = ROOT / "data" / "processed" / "labels_clean.parquet"
OUT_DIR = ROOT / "data" / "processed" / "images_256"
SIZE = 256
QUALITY = 90


def pad_to_square(img: Image.Image) -> Image.Image:
    """Put the image in the centre of a white square canvas."""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def main() -> None:                                   # NEW (everything below)
    labels = pl.read_parquet(LABELS)
    zf = zipfile.ZipFile(RAW_ZIP)
    in_zip = set(zf.namelist())

    done, skipped, missing = 0, 0, 0
    for aid in tqdm(labels["article_id"].to_list(), desc="Resizing"):
        zip_path = f"images/{aid[:3]}/{aid}.jpg"
        out_path = OUT_DIR / aid[:3] / f"{aid}.jpg"

        if out_path.exists():
            skipped += 1
            continue
        if zip_path not in in_zip:
            missing += 1
            continue

        with zf.open(zip_path) as f:
            img = Image.open(f)
            img.draft("RGB", (SIZE * 2, SIZE * 2))
            img = img.convert("RGB")
        small = pad_to_square(img).resize((SIZE, SIZE), Image.Resampling.LANCZOS)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        small.save(out_path, quality=QUALITY)
        done += 1

    print(f"Resized: {done} | already done: {skipped} | no image: {missing}")


if __name__ == "__main__":
    main()