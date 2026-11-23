"""Clean the H&M catalogue labels for the attribute classifier.

Applies the label decisions from reports/decisions.md (D3-D8):
- targets: product group (product_group_name) + colour (perceived_colour_master_name)
- drop 10 junk / out-of-scope product groups (incl. the 54 baby sleeping bags)
- drop products whose colour is "Unknown" / "undefined"
- merge two tiny colours into their nearest real colour

Input : data/raw/articles.csv
Output: data/processed/labels_clean.parquet
        columns: article_id (str, keeps leading zero), product_code, group, colour

Run from the project root:  python src/data/prepare_labels.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTICLES = ROOT / "data" / "raw" / "articles.csv"
OUT = ROOT / "data" / "processed" / "labels_clean.parquet"

# ---- Label decisions (see reports/decisions.md) ----
DROP_GROUPS = ["Unknown", "Underwear/nightwear", "Cosmetic", "Bags", "Items",
               "Furniture", "Garment and Shoe care", "Stationery",
               "Interior textile", "Fun"]                       # D4, D5
DROP_COLOURS = ["Unknown", "undefined"]                         # D6
COLOUR_MERGE = {"Yellowish Green": "Green", "Bluish Green": "Turquoise"}  # D6


def main() -> None:
    articles = pd.read_csv(ARTICLES, dtype={"article_id": str})

    labels = articles[["article_id", "product_code", "product_group_name",
                       "perceived_colour_master_name"]].copy()
    labels = labels.rename(columns={"product_group_name": "group",
                                    "perceived_colour_master_name": "colour"})
    labels["colour"] = labels["colour"].replace(COLOUR_MERGE)

    keep = ~labels["group"].isin(DROP_GROUPS) & ~labels["colour"].isin(DROP_COLOURS)
    print(f"Dropping {(~keep).sum():,} of {len(labels):,} products")
    labels = labels[keep].reset_index(drop=True)

    print(f"{labels['group'].nunique()} groups | {labels['colour'].nunique()} colours | "
          f"{len(labels):,} products kept")

    labels.to_parquet(OUT, index=False)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
