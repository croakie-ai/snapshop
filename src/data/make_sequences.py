"""Turn purchases into per-customer sequences for the recommender (Phase 6).

Time split (D9), no future information in training:
  train window : 2020-03-16 .. 2020-09-08  -> sequences the models learn from (6 months, recent fashion)
  val week     : 2020-09-09 .. 2020-09-15  -> pick the best epoch / settings
  test week    : 2020-09-16 .. 2020-09-22  -> final score, used once
Vocabulary: articles bought >= MIN_COUNT times in the train window (idx 0 = padding).

Makes data/processed/rec/:
  vocab.parquet       article_id, count, idx
  train_seqs.parquet  customer_id, items   (list of idx, time order, last MAX_LEN)
  val.parquet         customer_id, targets (article_ids bought in val week),  history (idx before it)
  test.parquet        customer_id, targets (article_ids bought in test week), history (idx before it)

Run from the project root:  python src/data/make_sequences.py
"""
from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
TX = ROOT / "data" / "processed" / "transactions.parquet"
OUT = ROOT / "data" / "processed" / "rec"

TRAIN_START = date(2020, 3, 16)
VAL_START, VAL_END = date(2020, 9, 9), date(2020, 9, 15)
TEST_START, TEST_END = date(2020, 9, 16), date(2020, 9, 22)
MIN_COUNT, MAX_LEN = 10, 50


def sequences(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """customer -> list of item idx in time order (most recent MAX_LEN)."""
    return (df.sort(["customer_id", "t_dat"])
              .group_by("customer_id", maintain_order=True)
              .agg(pl.col("idx").tail(MAX_LEN).alias(name)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tx = (pl.scan_parquet(TX)
            .filter(pl.col("t_dat") >= TRAIN_START)
            .select("t_dat", "customer_id", "article_id")
            .collect())
    print(f"{len(tx):,} purchases since {TRAIN_START}")

    vocab = (tx.filter(pl.col("t_dat") < VAL_START)
               .group_by("article_id").len("count")
               .filter(pl.col("count") >= MIN_COUNT)
               .sort("count", descending=True)
               .with_row_index("idx", offset=1))                   # 0 = padding
    vocab.write_parquet(OUT / "vocab.parquet")
    known = tx.join(vocab.select("article_id", "idx"), on="article_id", how="inner")
    print(f"vocab: {len(vocab):,} articles (bought >= {MIN_COUNT} times) cover {len(known):,} purchases")

    train = sequences(known.filter(pl.col("t_dat") < VAL_START), "items")
    train = train.filter(pl.col("items").list.len() >= 2)           # need at least 1 input + 1 target
    train.write_parquet(OUT / "train_seqs.parquet")
    print(f"train: {len(train):,} customers with 2+ purchases")

    for name, start, end in [("val", VAL_START, VAL_END), ("test", TEST_START, TEST_END)]:
        targets = (tx.filter(pl.col("t_dat").is_between(start, end))
                     .group_by("customer_id").agg(pl.col("article_id").unique().alias("targets")))
        history = sequences(known.filter(pl.col("t_dat") < start), "history")
        df = targets.join(history, on="customer_id", how="left")    # no history -> null (cold start)
        df.write_parquet(OUT / f"{name}.parquet")
        print(f"{name}: {len(df):,} customers bought something | {df['history'].null_count():,} have no history")


if __name__ == "__main__":
    main()