"""Convert the 3.5 GB purchases CSV into a fast-loading Parquet file.

- article_id read as text (keeps the leading zero, matches labels and image paths)
- t_dat parsed as a real Date (needed for the time-based recommender split)
- Parquet: ~7x smaller, loads in ~2 s instead of ~30 s, keeps column types

Input : data/raw/transactions_train.csv
Output: data/processed/transactions.parquet
        columns: t_dat (Date), customer_id (str), article_id (str), price (float, H&M-scaled),
                 sales_channel_id (1/2)

Run from the project root:  python src/data/prepare_transactions.py
"""
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "raw" / "transactions_train.csv"
OUT = ROOT / "data" / "processed" / "transactions.parquet"


def main() -> None:
    tx = pl.read_csv(CSV, schema_overrides={"article_id": pl.Utf8}, try_parse_dates=True)
    print(f"Rows: {tx.height:,} | dates {tx['t_dat'].min()} -> {tx['t_dat'].max()}")
    tx.write_parquet(OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
