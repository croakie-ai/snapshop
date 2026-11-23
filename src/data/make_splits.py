"""Leakage-safe, class-balanced catalogue splits (train / val / test).

- Groups by product_code, so all colour variants of a design land in the same split
  (otherwise near-identical images would sit in train AND test = data leakage).
- Stratifies by product group, so every class keeps roughly the same share in each split
  (so even the smallest class, Nightwear, has enough validation/test examples).
- Uses only products that have an image (decision D12).

Output: data/processed/splits.parquet
        columns: article_id, product_code, group, colour, split ("train" / "val" / "test")

Run from the project root:  python src/data/make_splits.py
"""
import zipfile
from pathlib import Path

import polars as pl
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
RAW_ZIP = ROOT / "data" / "raw" / "h-and-m-personalized-fashion-recommendations.zip"
LABELS = ROOT / "data" / "processed" / "labels_clean.parquet"
OUT = ROOT / "data" / "processed" / "splits.parquet"

SEED = 42      # fixed randomness -> identical split on every run (reproducibility)
N_FOLDS = 7    # 7 equal slices: 1 test, 1 val, 5 train  (~71% / 14% / 14%)


def products_with_images() -> set[str]:
    """article_ids that have a photo inside the H&M zip."""
    with zipfile.ZipFile(RAW_ZIP) as zf:
        return {
            Path(name).stem
            for name in zf.namelist()
            if name.startswith("images/") and name.endswith(".jpg")
        }


def assign_splits(labels: pl.DataFrame) -> pl.DataFrame:
    """Add a 'split' column using a stratified, grouped split."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = sgkf.split(
        X=labels["article_id"].to_numpy(),
        y=labels["group"].to_numpy(),           # balance this (the class)
        groups=labels["product_code"].to_numpy(),  # keep these together (the design)
    )
    _, test_idx = next(folds)   # first slice  -> test
    _, val_idx = next(folds)    # second slice -> val (never overlaps the first)

    split = ["train"] * labels.height
    for i in test_idx:
        split[i] = "test"
    for i in val_idx:
        split[i] = "val"
    return labels.with_columns(pl.Series("split", split))


def check_no_leakage(labels: pl.DataFrame) -> int:
    """Number of designs whose variants ended up in more than one split (must be 0)."""
    return (
        labels.group_by("product_code")
        .agg(pl.col("split").n_unique().alias("n_splits"))
        .filter(pl.col("n_splits") > 1)
        .height
    )


def class_balance_table(labels: pl.DataFrame) -> pl.DataFrame:
    """Products per class per split, plus each split's share of that class."""
    counts = (
        labels.group_by("group", "split").len()
        .pivot(on="split", index="group", values="len")
        .fill_null(0)
    )
    total = pl.col("train") + pl.col("val") + pl.col("test")
    return counts.with_columns(
        (pl.col("val") / total * 100).round(1).alias("val_%"),
        (pl.col("test") / total * 100).round(1).alias("test_%"),
    ).sort("train", descending=True)


def main() -> None:
    labels = pl.read_parquet(LABELS)
    labels = labels.filter(pl.col("article_id").is_in(list(products_with_images())))
    print(f"Products with images: {labels.height:,}")

    labels = assign_splits(labels)

    print("\nSplit sizes:")
    print(labels.group_by("split").len().sort("split"))

    leaks = check_no_leakage(labels)
    print(f"\nDesigns appearing in more than one split: {leaks}  (must be 0)")
    if leaks:
        raise RuntimeError("Leakage detected: a product_code spans several splits.")

    pl.Config.set_tbl_rows(20)
    print("\nClass balance (products per product group):")
    print(class_balance_table(labels))

    labels.write_parquet(OUT)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
