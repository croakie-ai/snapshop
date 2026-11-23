"""Build a small 32x32 RGB dataset for the from-scratch NumPy network (Phase 2).

Why 32x32?
- The NumPy MLP takes a flat vector, so the input size sets the size of its first layer:
  32*32*3 = 3,072 inputs (vs 224*224*3 = 150,528, far too big for a CPU-only NumPy network).
- Colour is kept (RGB), so the same network can be a baseline for BOTH targets:
  product group (9 classes) and colour (16 classes).
- It is the "no convolution" baseline the Phase 3 CNN has to beat.

Uses the same train/val/test split as everything else (splits.parquet), so the
comparison with the CNN is fair. Test arrays are stored but must not be looked at
until the final evaluation.

Output: data/processed/mlp_32x32.npz
        X_{split}        : uint8, shape (n, 3072), pixel values 0-255, laid out as 32x32x3
        y_group_{split}  : int64 product-group index (0-8)
        y_colour_{split} : int64 colour index (0-15)
        ids_{split}      : article_id of each row (for the error gallery)
        group_classes, colour_classes : class names in index order

Run from the project root:  python src/data/make_mlp_dataset.py
"""
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "processed" / "splits.parquet"
IMG_DIR = ROOT / "data" / "processed" / "images_256"
OUT = ROOT / "data" / "processed" / "mlp_32x32.npz"
SIZE = 32


def load_small_rgb(article_id: str) -> np.ndarray:
    """Open a cached 256px image and return it as a 32x32x3 uint8 array."""
    img = Image.open(IMG_DIR / article_id[:3] / f"{article_id}.jpg")
    img.draft("RGB", (SIZE * 2, SIZE * 2))          # fast reduced-size JPEG decode
    img = img.convert("RGB").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)           # shape (32, 32, 3)


def main() -> None:
    df = pl.read_parquet(SPLITS)
    group_classes = sorted(df["group"].unique().to_list())
    colour_classes = sorted(df["colour"].unique().to_list())
    group_idx = {name: i for i, name in enumerate(group_classes)}
    colour_idx = {name: i for i, name in enumerate(colour_classes)}
    print("Groups :", group_classes)
    print("Colours:", colour_classes)

    arrays = {}
    for split in ["train", "val", "test"]:
        part = df.filter(pl.col("split") == split)
        ids = part["article_id"].to_list()

        X = np.empty((len(ids), SIZE * SIZE * 3), dtype=np.uint8)
        for i, aid in enumerate(tqdm(ids, desc=split)):
            X[i] = load_small_rgb(aid).reshape(-1)        # 32x32x3 -> 3,072 values

        arrays[f"X_{split}"] = X
        arrays[f"y_group_{split}"] = np.array(
            [group_idx[g] for g in part["group"].to_list()], dtype=np.int64)
        arrays[f"y_colour_{split}"] = np.array(
            [colour_idx[c] for c in part["colour"].to_list()], dtype=np.int64)
        arrays[f"ids_{split}"] = np.array(ids)
        print(f"{split}: X {X.shape}")

    np.savez_compressed(
        OUT,
        group_classes=np.array(group_classes),
        colour_classes=np.array(colour_classes),
        **arrays,
    )
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
