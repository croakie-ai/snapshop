"""Convert a DeepFashion2 subset to YOLO-seg format (Phase 5).

DeepFashion2 gives one .json per image (polygons in pixels, categories 1-13).
YOLO-seg wants: images/<split>/x.jpg + labels/<split>/x.txt, one line per garment:
    class x1 y1 x2 y2 ...   (outline points, divided by image width/height -> 0 to 1)

Subsets (seed 42):
  train: 30,000 random DF2 train images        -> YOLO trains on these
  val  :  2,000 random DF2 validation images   -> YOLO picks its best epoch on these
  The other ~30k DF2 validation images stay untouched for our final evaluation.

Run from the project root:  python src/data/make_yolo_df2.py
"""
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics.data.converter import merge_multi_segment

ROOT = Path(__file__).resolve().parents[2]
DF2 = ROOT / "data" / "raw" / "deepfashion2"
OUT = ROOT / "data" / "processed" / "df2_yolo"
N_TRAIN, N_VAL, SEED = 30_000, 2_000, 42

# DeepFashion2 category_id 1..13  ->  YOLO class 0..12
NAMES = ["short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear",
         "vest", "sling", "shorts", "trousers", "skirt", "short sleeve dress",
         "long sleeve dress", "vest dress", "sling dress"]


def to_yolo_lines(anno: dict, w: int, h: int) -> list[str]:
    """One YOLO line per garment: class + outline points scaled to 0-1."""
    lines = []
    for key, item in anno.items():
        if not key.startswith("item"):                       # skip "source", "pair_id"
            continue
        polys = [np.array(p, dtype=float).reshape(-1, 2) for p in item["segmentation"] if len(p) >= 6]
        if not polys:
            continue
        pts = np.concatenate(merge_multi_segment(polys), axis=0) if len(polys) > 1 else polys[0]
        pts = (pts / [w, h]).clip(0, 1)
        lines.append(" ".join([str(item["category_id"] - 1)] + [f"{v:.5f}" for v in pts.ravel()]))
    return lines


def convert(src_split: str, dst_split: str, n: int, rng: random.Random) -> None:
    ids = sorted(p.stem for p in (DF2 / src_split / "image").glob("*.jpg"))
    chosen = rng.sample(ids, n)
    (OUT / "images" / dst_split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / dst_split).mkdir(parents=True, exist_ok=True)
    empty = 0
    for i in tqdm(chosen, desc=dst_split):
        img_src = DF2 / src_split / "image" / f"{i}.jpg"
        anno = json.loads((DF2 / src_split / "annos" / f"{i}.json").read_text())
        w, h = Image.open(img_src).size                       # reads only the file header: fast
        lines = to_yolo_lines(anno, w, h)
        empty += not lines
        shutil.copy2(img_src, OUT / "images" / dst_split / f"{i}.jpg")
        (OUT / "labels" / dst_split / f"{i}.txt").write_text("\n".join(lines))
    (OUT / f"{dst_split}_ids.txt").write_text("\n".join(chosen))   # saved list: reproducible, no leakage later
    print(f"{dst_split}: {n:,} images, {empty} without garments")


def main() -> None:
    rng = random.Random(SEED)
    convert("train", "train", N_TRAIN, rng)
    convert("validation", "val", N_VAL, rng)
    yaml = [f"path: {OUT.as_posix()}", "train: images/train", "val: images/val", "names:"]
    yaml += [f"  {i}: {name}" for i, name in enumerate(NAMES)]
    (OUT / "df2.yaml").write_text("\n".join(yaml))
    print("Wrote", OUT / "df2.yaml")


if __name__ == "__main__":
    main()