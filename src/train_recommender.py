"""Next-purchase recommender (Phase 6): LSTM or Transformer over purchase "sentences".

Each customer's purchases (item numbers, time order) = a sentence. At every position the model
predicts the NEXT item (cross-entropy over all ~30k items, like a language model). Same inputs,
same training, same evaluation for both models, so the comparison is fair (FAQ Q13).

  item embeddings : 64 numbers per product, shared with the output layer ("weight tying")
  lstm            : 1 layer, reads left to right with a memory vector
  transformer     : 2 layers, 2 attention heads, learned position embeddings, CAUSAL mask
  output          : score for every product = summary . product embedding -> top 12

Model selection: MAP@12 on the val week (9-15 Sep 2020) after every epoch; the test week is untouched.
Customers with no history get popularity (cold start), exactly like the baselines (notebook 08).

Makes: runs/rec_<model>.pt (best epoch), runs/rec_<model>_history.json
Run from the project root:  python src/train_recommender.py --model lstm      (then --model transformer)
"""
import argparse
import json
import random
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
REC, RUNS = ROOT / "data" / "processed" / "rec", ROOT / "runs"
DIM, MAX_LEN, BATCH, LR = 64, 50, 256, 1e-3
EPOCHS, PATIENCE, SEED = 10, 2, 42


class Recommender(nn.Module):
    def __init__(self, n_items: int, kind: str):
        super().__init__()
        self.kind = kind
        self.items = nn.Embedding(n_items, DIM, padding_idx=0)
        if kind == "lstm":
            self.seq = nn.LSTM(DIM, DIM, batch_first=True)
        else:
            self.pos = nn.Embedding(MAX_LEN, DIM)
            layer = nn.TransformerEncoderLayer(DIM, nhead=2, dim_feedforward=4 * DIM, dropout=0.1,
                                               batch_first=True, norm_first=True)
            self.seq = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(DIM)

    def forward(self, x):
        """x: (batch, length) item numbers, 0 = padding -> a 64-number summary at every position."""
        h = self.items(x)
        if self.kind == "lstm":
            h, _ = self.seq(h)
        else:
            L = x.shape[1]
            h = h + self.pos(torch.arange(L, device=x.device))
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
            h = self.seq(h, mask=causal, src_key_padding_mask=(x == 0))   # True = may NOT look there
        return self.norm(h)

    def scores(self, h):
        return h @ self.items.weight.T          # one score per product (weight tying)


def pad(seqs) -> torch.Tensor:
    out = torch.zeros(len(seqs), max(len(s) for s in seqs), dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s)
    return out


def apk(actual, predicted, k=12) -> float:
    if not actual:
        return 0.0
    hits, score = 0, 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual), k)


@torch.no_grad()
def recommend(model, histories, device, k=12):
    """Top-k item numbers after each (non-empty) history."""
    model.eval()
    recs = []
    for i in range(0, len(histories), 512):
        chunk = [h[-MAX_LEN:] for h in histories[i:i + 512]]
        x = pad(chunk).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            h = model(x)
        last = torch.tensor([len(c) - 1 for c in chunk], device=device)
        s = model.scores(h[torch.arange(len(chunk), device=device), last].float())
        s[:, 0] = -1e9                                                  # never recommend "padding"
        recs += s.topk(k, dim=1).indices.cpu().tolist()
    return recs


def map12(model, df, pop, idx2art, device, last_only=False) -> float:
    targets, hists = df["targets"].to_list(), df["history"].to_list()
    known = [i for i, h in enumerate(hists) if h]
    inputs = [hists[i][-1:] if last_only else hists[i] for i in known]
    recs = dict(zip(known, recommend(model, inputs, device)))
    return float(np.mean([apk(set(t), [idx2art[j] for j in recs[i]] if i in recs else pop)
                          for i, t in enumerate(targets)]))


def popular(start: date, end: date, k=12):
    return (pl.scan_parquet(ROOT / "data" / "processed" / "transactions.parquet")
              .filter(pl.col("t_dat").is_between(start, end))
              .group_by("article_id").len().sort("len", descending=True)
              .head(k).collect()["article_id"].to_list())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "transformer"], required=True)
    args = ap.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = pl.read_parquet(REC / "vocab.parquet")
    idx2art = dict(zip(vocab["idx"].to_list(), vocab["article_id"].to_list()))
    n_items = len(vocab) + 1                                            # + padding (0)
    seqs = [s[-MAX_LEN:] for s in pl.read_parquet(REC / "train_seqs.parquet")["items"].to_list()]
    val = pl.read_parquet(REC / "val.parquet")
    pop = popular(date(2020, 9, 2), date(2020, 9, 8))

    model = Recommender(n_items, args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()):,} parameters | "
          f"{len(seqs):,} training sequences | {n_items - 1:,} products")

    history, best, bad = [], -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        random.shuffle(seqs)
        t0, running, n = time.time(), 0.0, 0
        for i in tqdm(range(0, len(seqs), BATCH), desc=f"epoch {epoch}"):
            x = pad(seqs[i:i + BATCH]).to(device)
            inp, tgt = x[:, :-1], x[:, 1:]                              # predict the NEXT item everywhere
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                h = model(inp)
            keep = tgt > 0                                              # skip padding positions
            loss = F.cross_entropy(model.scores(h[keep].float()), tgt[keep])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            n += 1

        full = map12(model, val, pop, idx2art, device)
        last = map12(model, val, pop, idx2art, device, last_only=True)
        history.append({"epoch": epoch, "train_loss": running / n, "val_map12_full": full,
                        "val_map12_last_item": last, "minutes": round((time.time() - t0) / 60, 1)})
        (RUNS / f"rec_{args.model}_history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch {epoch} | loss {running / n:.3f} | val MAP@12 full {full:.4f} | last item {last:.4f}")

        if full > best:
            best, bad = full, 0
            torch.save(model.state_dict(), RUNS / f"rec_{args.model}.pt")
            print("  new best: weights saved")
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"No improvement for {PATIENCE} epochs, stopping.")
                break


if __name__ == "__main__":
    main()
