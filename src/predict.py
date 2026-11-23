"""SnapShop on ANY photo: find each garment (YOLO), then tags (output 2) + top-10 similar H&M products (output 1).

Step 0 (Phase 5, D25): YOLO11s-seg finds the garments. Each detection with confidence >= 0.5 (max 3) is
cut out with its BOX (street-to-shop test: box crops = mask crops for search, and boxes avoid the
trousers/shorts mask issue). No confident detection (e.g. a bag, a clean product shot) -> the full photo.
Every crop then gets exactly the same preparation as the training images:
  white-pad to square -> resize 256 -> centre-crop 224 -> ImageNet normalize
Then two models (D24):
  ft_ls  (fine-tuned ResNet-50 + label smoothing): group + colour TAGS, and its 2,048-d colour/type fingerprint
  metric (ResNet-50 + ArcFace):                    512-d DESIGN fingerprint
Search: hybrid score = alpha * cos(design) + (1 - alpha) * cos(colour/type), against all 104k catalogue
products (exact cosine search with FAISS, same as notebook 07).
--match: design (alpha 1.0), balanced (0.5, default), colour (0.25)

Step 3 (Phase 6): "buy next" -- the first of the top-10 matches that the recommender knows becomes the
input; the recommender (transformer if trained, else LSTM) predicts the 12 most likely next purchases.
If none of the matches is known (rarely bought items) -> the latest week's best-sellers (cold start).

Saves a picture of the results next to the photo: <photo name>_snapshop.png (one row per garment)

Run from the project root:
    python src/predict.py data/street_test/test1.jpg
    python src/predict.py data/street_test/test1.jpg --match design --k 10
    python src/predict.py data/street_test/test1.jpg --no-detect          (skip YOLO, use the full photo)
"""
import argparse
from datetime import date
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from ultralytics import YOLO

from data.images import IMG_DIR, class_lists, get_transforms
from data.make_images import SIZE, pad_to_square
from train_finetune import MultiHeadResNet
from train_metric import Embedder
from train_recommender import Recommender, popular, recommend

ROOT = Path(__file__).resolve().parents[1]
RUNS, PROC = ROOT / "runs", ROOT / "data" / "processed"
ALPHA = {"design": 1.0, "balanced": 0.5, "colour": 0.25}
YOLO_W = RUNS / "segment" / "runs" / "yolo" / "df2_y11s" / "weights" / "best.pt"
MIN_CONF, MAX_GARMENTS, MARGIN = 0.5, 3, 0.05
TAG_TRUST, SEARCH_POOL = 0.6, 200
REC_DIR = PROC / "rec"
ENABLE_RECS = True           # output 3 (buy next)
REC_MODEL = "lstm"           # chosen on val MAP@12 (D27): LSTM 0.0217 vs transformer 0.0215
# YOLO's garment type (DeepFashion2) -> H&M product group. YOLO was trained on WORN clothes, so on
# street photos its type is more reliable than the tagger (trained on flat product shots) (D26)
GROUP_OF = {**dict.fromkeys(["short sleeve top", "long sleeve top", "short sleeve outwear",
                             "long sleeve outwear", "vest", "sling"], "Garment Upper body"),
            **dict.fromkeys(["shorts", "trousers", "skirt"], "Garment Lower body"),
            **dict.fromkeys(["short sleeve dress", "long sleeve dress", "vest dress", "sling dress"],
                            "Garment Full body")}


def load_models(device: str):
    tagger = MultiHeadResNet()                                    # Phase 3 best (run B)
    tagger.load_state_dict(torch.load(RUNS / "finetuned_resnet50_ls.pt", map_location="cpu"))
    designer = Embedder()                                         # Phase 4 ArcFace model
    designer.load_state_dict(torch.load(RUNS / "metric_arcface.pt", map_location="cpu"))
    return tagger.to(device).eval(), designer.to(device).eval()


def load_photo(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")   # phone photos: apply the stored rotation


def detect(photo: Image.Image, yolo) -> list:
    """YOLO -> [(crop, label, H&M group hint)] for each confident garment (max 3); the full photo if none."""
    r = yolo.predict(photo, imgsz=640, conf=MIN_CONF, verbose=False)[0]
    crops = []
    for (x1, y1, x2, y2), conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        mx, my = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN            # small margin around the box
        box = (max(0, x1 - mx), max(0, y1 - my), min(photo.width, x2 + mx), min(photo.height, y2 + my))
        name = r.names[int(cls)]
        crops.append((photo.crop(box), f"{name} ({conf:.2f})", GROUP_OF.get(name)))
    return crops[:MAX_GARMENTS] or [(photo, "full photo (no garment detected)", None)]


def prepare(img: Image.Image) -> torch.Tensor:
    img = pad_to_square(img).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    return get_transforms(train=False)(img).unsqueeze(0)        # (1, 3, 224, 224): a batch of one


@torch.no_grad()
def run_models(tagger, designer, x, device: str):
    x = x.to(device)
    feats = tagger.backbone(x)                                    # 2,048 numbers: shared by tags + colour fingerprint
    out_g, out_c = tagger.group_head(feats), tagger.colour_head(feats)
    colour_fp = F.normalize(feats, dim=1)[0].cpu().numpy()        # length 1, like the catalogue fingerprints
    design_fp = designer(x)[0].cpu().numpy()                      # already length 1
    return out_g, out_c, colour_fp, design_fp


def tags(out_g, out_c, top: int) -> dict:
    groups, colours = class_lists()
    result = {}
    for task, out, names in [("group", out_g, groups), ("colour", out_c, colours)]:
        conf, idx = out.softmax(dim=1)[0].cpu().topk(top)
        result[task] = [(names[i], float(c)) for c, i in zip(conf, idx)]
    return result


def final_group(group_preds, hint):
    """Tagger's group, unless it disagrees with YOLO's garment type or is unsure (< 60%) -> YOLO's group."""
    name, conf = group_preds[0]
    if hint and (name != hint or conf < TAG_TRUST):
        return hint, "from detector"
    return name, f"{conf * 100:.0f}%"


def load_recommender(device: str):
    """The chosen next-purchase model (REC_MODEL) + item number <-> article id."""
    vocab = pl.read_parquet(REC_DIR / "vocab.parquet")
    art2idx = dict(zip(vocab["article_id"].to_list(), vocab["idx"].to_list()))
    idx2art = {i: a for a, i in art2idx.items()}
    pop = popular(date(2020, 9, 16), date(2020, 9, 22))              # latest week in the data = "trending now"
    for kind in [REC_MODEL]:
        w = RUNS / f"rec_{kind}.pt"
        if w.exists():
            model = Recommender(len(vocab) + 1, kind)
            model.load_state_dict(torch.load(w, map_location="cpu"))
            return {"model": model.to(device).eval(), "kind": kind, "art2idx": art2idx, "idx2art": idx2art, "pop": pop}
    return {"model": None, "kind": "popularity only (no recommender trained yet)", "art2idx": art2idx,
            "idx2art": idx2art, "pop": pop}


def buy_next(rec: dict, matches: list, device: str, k: int = 12):
    """The first search match the recommender knows -> its k most likely next purchases."""
    seed = next((a for a in matches if a in rec["art2idx"]), None)
    if rec["model"] is None or seed is None:
        return rec["pop"][:k], None
    items = recommend(rec["model"], [[rec["art2idx"][seed]]], device, k + 1)[0]
    arts = [rec["idx2art"][i] for i in items if rec["idx2art"].get(i) != seed]
    return arts[:k], seed


def build_index(alpha: float):
    """Catalogue fingerprints (extract_embeddings.py), weighted exactly like the query."""
    d, c = torch.load(PROC / "emb_metric.pt"), torch.load(PROC / "emb_ft_ls.pt")
    assert d["ids"] == c["ids"]                                   # same product order in both files
    X = np.hstack([d["emb"].float().numpy() * np.sqrt(alpha),
                   c["emb"].float().numpy() * np.sqrt(1 - alpha)])
    index = faiss.IndexFlatIP(X.shape[1])                         # inner product of length-1 parts = cosine
    index.add(X)
    return index, d                                               # d also holds ids, group, colour per product


def search(index, design_fp, colour_fp, alpha: float, k: int, meta=None, hint=None):
    """Hybrid cosine search. With a group hint: products of that group first (in score order), then the rest."""
    q = np.hstack([design_fp * np.sqrt(alpha), colour_fp * np.sqrt(1 - alpha)])[None].astype("float32")
    scores, idx = index.search(q, SEARCH_POOL if hint else k)
    scores, idx = scores[0], idx[0]
    if hint:
        same = [i for i, j in enumerate(idx) if meta["group"][j] == hint]
        order = same + [i for i in range(len(idx)) if i not in set(same)]
        scores, idx = scores[order], idx[order]
    return scores[:k], idx[:k]


def save_gallery(rows: list, meta: dict, out_path: Path, title: str) -> None:
    """rows: [(crop image, crop title, scores, idx)] -> one row per garment: the crop + its top-k."""
    k = len(rows[0][3])
    fig, axes = plt.subplots(len(rows), k + 1, figsize=(1.9 * (k + 1), 3.3 * len(rows)), squeeze=False)
    for r, (crop, crop_title, scores, idx) in enumerate(rows):
        axes[r, 0].imshow(crop)
        axes[r, 0].set_title(crop_title, fontsize=8, weight="bold")
        for rank, (ax, s, j) in enumerate(zip(axes[r, 1:], scores, idx), start=1):
            aid = meta["ids"][j]
            ax.imshow(Image.open(IMG_DIR / aid[:3] / f"{aid}.jpg"))
            ax.set_title(f"#{rank}  {meta['group'][j].replace('Garment ', '')}\n{meta['colour'][j]}  ({s:.2f})", fontsize=7)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(title, fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="SnapShop: find garments, then tags + similar H&M products for any photo")
    ap.add_argument("image", type=Path)
    ap.add_argument("--match", choices=list(ALPHA), default="balanced", help="design / balanced / colour")
    ap.add_argument("--k", type=int, default=10, help="how many similar products")
    ap.add_argument("--top", type=int, default=3, help="how many guesses per tag")
    ap.add_argument("--no-detect", action="store_true", help="skip YOLO and use the full photo")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha = ALPHA[args.match]
    photo = load_photo(args.image)
    garments = [(photo, "full photo", None)] if args.no_detect else detect(photo, YOLO(str(YOLO_W)))
    tagger, designer = load_models(device)
    index, meta = build_index(alpha)
    rec = load_recommender(device) if ENABLE_RECS else None

    print(f"\n{args.image.name}: {len(garments)} garment(s)  [match = {args.match}, alpha = {alpha}]")
    rows = []
    for n, (crop, label, hint) in enumerate(garments, start=1):
        out_g, out_c, colour_fp, design_fp = run_models(tagger, designer, prepare(crop), device)
        print(f"\n  garment {n}: {label}")
        t = tags(out_g, out_c, args.top)
        for task, preds in t.items():
            print(f"    {task}: " + ", ".join(f"{name} {c * 100:.1f}%" for name, c in preds))
        group, why = final_group(t["group"], hint)
        print(f"    => final tags: {group} ({why}) | {t['colour'][0][0]}")
        scores, idx = search(index, design_fp, colour_fp, alpha, args.k, meta, hint)
        print(f"    top-{args.k} similar H&M products:")
        for rank, (s_, j) in enumerate(zip(scores, idx), start=1):
            print(f"      #{rank:<2} {meta['ids'][j]}  {meta['group'][j]:<20} {meta['colour'][j]:<13} score {s_:.3f}")
        rows.append((crop, f"garment {n}: {label}", scores, idx))
        if rec is not None:
            recs, seed = buy_next(rec, [meta["ids"][j] for j in idx], device)
            print(f"    buy next ({rec['kind']}, from {seed or 'trending now'}): " + ", ".join(recs))

    out_path = args.image.with_name(f"{args.image.stem}_snapshop.png")
    save_gallery(rows, meta, out_path, f"SnapShop top-{args.k}  (match = {args.match})")
    print(f"\n  picture saved: {out_path}")


if __name__ == "__main__":
    main()
