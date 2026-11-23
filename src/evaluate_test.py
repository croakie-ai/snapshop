"""FINAL TEST-SET EVALUATION — run ONCE, after every model choice was made on validation data.

1. Classifier (ft_ls)   : H&M test split (14,860 products)  -> accuracy + macro-F1, group and colour
2. Visual search        : test products with a sibling as queries, whole 104k catalogue as gallery
                          -> Recall@1/5/10/20 + group/colour match@10 for ImageNet, metric, and the final hybrid
3. Recommender          : test week 16-22 Sep 2020 -> MAP@12 (full history / last item only)
                          popularity, co-purchase, LSTM, transformer (whichever are trained)
(YOLO's test score and the street-to-shop test are in D25: they already used untouched data.)

Makes: runs/test_results.json  (+ printed tables)
Run from the project root:  python src/evaluate_test.py
"""
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import faiss
import numpy as np
import polars as pl
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.images import ProductImages
from train_finetune import MultiHeadResNet
from train_recommender import MAX_LEN, Recommender, apk, map12, popular

ROOT = Path(__file__).resolve().parents[1]
RUNS, PROC = ROOT / "runs", ROOT / "data" / "processed"
device = "cuda" if torch.cuda.is_available() else "cpu"
results = {}


@torch.no_grad()
def test_classifier():
    model = MultiHeadResNet()
    model.load_state_dict(torch.load(RUNS / "finetuned_resnet50_ls.pt", map_location="cpu"))
    model = model.to(device).eval()
    ds = ProductImages("test")
    pg, pc = [], []
    for x, _, _ in tqdm(DataLoader(ds, batch_size=128, num_workers=2), desc="classifier (test)"):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            og, oc = model(x.to(device))
        pg.append(og.argmax(1).cpu()); pc.append(oc.argmax(1).cpu())
    pg, pc = torch.cat(pg).numpy(), torch.cat(pc).numpy()
    yg, yc = np.array(ds.y_group), np.array(ds.y_colour)
    r = {"n": len(yg),
         "group_acc": float((pg == yg).mean()), "group_macro_f1": float(f1_score(yg, pg, average="macro")),
         "colour_acc": float((pc == yc).mean()), "colour_macro_f1": float(f1_score(yc, pc, average="macro"))}
    results["classifier_ft_ls"] = r
    print(f"\nCLASSIFIER (test, {r['n']:,} products): group {r['group_acc']:.3f} / F1 {r['group_macro_f1']:.3f} | "
          f"colour {r['colour_acc']:.3f} / F1 {r['colour_macro_f1']:.3f}")


def test_search():
    data = {m: torch.load(PROC / f"emb_{m}.pt") for m in ["imagenet", "metric", "ft_ls"]}
    d0 = data["metric"]
    code, split = np.array(d0["product_code"]), np.array(d0["split"])
    group, colour = np.array(d0["group"]), np.array(d0["colour"])
    _, inv, counts = np.unique(code, return_inverse=True, return_counts=True)
    queries = np.where((split == "test") & (counts[inv] > 1))[0]

    def run(X):
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        _, nb = index.search(X[queries], 21)
        nb = np.array([row[row != q][:20] for row, q in zip(nb, queries)])
        same = code[nb] == code[queries][:, None]
        out = {f"recall@{k}": float(same[:, :k].any(1).mean()) for k in [1, 5, 10, 20]}
        out["group_match@10"] = float((group[nb[:, :10]] == group[queries][:, None]).mean())
        out["colour_match@10"] = float((colour[nb[:, :10]] == colour[queries][:, None]).mean())
        return out

    f = {m: data[m]["emb"].float().numpy() for m in data}
    setups = {"imagenet (baseline)": f["imagenet"], "metric (ArcFace)": f["metric"],
              "hybrid metric + ft_ls, alpha 0.5 (final)": np.hstack([f["metric"] * np.sqrt(0.5), f["ft_ls"] * np.sqrt(0.5)])}
    results["search"] = {"n_queries": int(len(queries))}
    print(f"\nSEARCH (test, {len(queries):,} queries vs all {len(code):,} products)")
    for name, X in setups.items():
        r = run(X)
        results["search"][name] = r
        print(f"  {name:<42} R@1 {r['recall@1']:.3f}  R@10 {r['recall@10']:.3f}  R@20 {r['recall@20']:.3f}  "
              f"group {r['group_match@10']:.3f}  colour {r['colour_match@10']:.3f}")


def test_recommender():
    rec = PROC / "rec"
    vocab = pl.read_parquet(rec / "vocab.parquet")
    idx2art = dict(zip(vocab["idx"].to_list(), vocab["article_id"].to_list()))
    test = pl.read_parquet(rec / "test.parquet")
    pop = popular(date(2020, 9, 9), date(2020, 9, 15))                  # the week right before the test week
    targets, hists = test["targets"].to_list(), test["history"].to_list()

    co = defaultdict(Counter)
    for items in pl.read_parquet(rec / "train_seqs.parquet")["items"].to_list():
        for i, a in enumerate(items):
            for b in items[i + 1:i + 4]:
                if b != a:
                    co[a][b] += 1

    def copurchase(h):
        if not h:
            return pop
        s = Counter()
        for w, a in enumerate(reversed(h[-5:])):
            for b, c in co[a].most_common(50):
                s[b] += c / (w + 1)
        r = [idx2art[b] for b, _ in s.most_common(12)]
        return (r + [p for p in pop if p not in r])[:12]

    def score(fn, last_only):
        return float(np.mean([apk(set(t), fn((h or [])[-1:] if last_only else (h or []))) for t, h in zip(targets, hists)]))

    out = {"n_customers": len(test),
           "popularity": {"full": score(lambda h: pop, False), "last_item": score(lambda h: pop, True)},
           "co-purchase": {"full": score(copurchase, False), "last_item": score(copurchase, True)}}
    for kind in ["lstm", "transformer"]:
        w = RUNS / f"rec_{kind}.pt"
        if w.exists():
            m = Recommender(len(vocab) + 1, kind)
            m.load_state_dict(torch.load(w, map_location="cpu"))
            m = m.to(device)
            out[kind] = {"full": map12(m, test, pop, idx2art, device),
                         "last_item": map12(m, test, pop, idx2art, device, last_only=True)}
    results["recommender_map12"] = out
    print(f"\nRECOMMENDER (test week, {len(test):,} customers)  MAP@12   full history | last item only")
    for k, v in out.items():
        if isinstance(v, dict):
            print(f"  {k:<14} {v['full']:.4f} | {v['last_item']:.4f}")


if __name__ == "__main__":
    test_classifier()
    test_search()
    test_recommender()
    (RUNS / "test_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {RUNS / 'test_results.json'}")
