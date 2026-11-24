"""One clean picture of SnapShop working on a photo (for the README / a LinkedIn post).

Left: the photo with YOLO's boxes. Then one row per garment: the crop + its tags, the 6 most similar
H&M products, and 5 "buy next" picks from the LSTM. Underwear / swimwear / nightwear are left out.

Run from the project root:
    python src/showcase.py data/street_test/mine/outfit.jpg       -> reports/figures/showcase_outfit.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from PIL import Image
from ultralytics import YOLO

from data.images import IMG_DIR
from predict import (ALPHA, GROUP_OF, MARGIN, MAX_GARMENTS, MIN_CONF, ROOT, SEARCH_POOL, YOLO_W, build_index,
                     buy_next, final_group, load_models, load_photo, load_recommender, prepare, run_models,
                     search, tags)

HIDE = {"Underwear", "Swimwear", "Nightwear"}
N_SIM, N_REC = 6, 5
ACCENT = "#D9622B"


def short(group: str) -> str:
    return group.replace("Garment ", "")


def has_photo(aid: str) -> bool:
    return (IMG_DIR / aid[:3] / f"{aid}.jpg").exists()


def product_photo(aid: str) -> Image.Image:
    return Image.open(IMG_DIR / aid[:3] / f"{aid}.jpg").convert("RGB")


def detect(photo, yolo):
    """Same rules as predict.detect, plus the photo with YOLO's boxes drawn on it."""
    r = yolo.predict(photo, imgsz=640, conf=MIN_CONF, verbose=False)[0]
    drawn = Image.fromarray(np.ascontiguousarray(r.plot(masks=False)[..., ::-1]))
    crops = []
    for (x1, y1, x2, y2), conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        mx, my = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
        box = (max(0, x1 - mx), max(0, y1 - my), min(photo.width, x2 + mx), min(photo.height, y2 + my))
        name = r.names[int(cls)]
        crops.append((photo.crop(box), name, GROUP_OF.get(name)))
    return crops[:MAX_GARMENTS] or [(photo, "full photo", None)], drawn


def analyse(photo, device, match="balanced"):
    yolo = YOLO(str(YOLO_W))
    tagger, designer = load_models(device)
    alpha = ALPHA[match]
    index, meta = build_index(alpha)
    info = dict(zip(meta["ids"], meta["group"]))
    rec = load_recommender(device)

    garments, drawn = detect(photo, yolo)
    rows = []
    for crop, name, hint in garments:
        out_g, out_c, colour_fp, design_fp = run_models(tagger, designer, prepare(crop), device)
        t = tags(out_g, out_c, 1)
        group, _ = final_group(t["group"], hint)
        _, idx = search(index, design_fp, colour_fp, alpha, SEARCH_POOL, meta, hint)
        similar = [meta["ids"][j] for j in idx if meta["group"][j] not in HIDE and has_photo(meta["ids"][j])][:N_SIM]
        recs, _ = buy_next(rec, similar, device, k=40)
        recs = [a for a in recs if a in info and info[a] not in HIDE and has_photo(a)][:N_REC]
        rows.append({"crop": crop, "label": f"{short(group)} · {t['colour'][0][0]}", "similar": similar, "recs": recs})
    return drawn, rows


def draw(drawn, rows, out_path):
    n = len(rows)
    cols = 1 + 1 + N_SIM + 1 + N_REC                       # photo | crop | similar | gap | buy next
    widths = [3.2, 1.6] + [1.3] * N_SIM + [0.25] + [1.3] * N_REC
    fig = plt.figure(figsize=(sum(widths) * 1.15, 2.35 * n + 1.1))
    gs = GridSpec(n, cols, figure=fig, width_ratios=widths, wspace=0.08, hspace=0.45,
                  left=0.01, right=0.99, top=1 - 0.95 / (2.35 * n + 1.1), bottom=0.02)

    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(drawn)
    ax.set_title("photo → garments found (YOLO11)", fontsize=11, weight="bold", loc="left")
    ax.axis("off")

    for r, row in enumerate(rows):
        ax = fig.add_subplot(gs[r, 1])
        ax.imshow(row["crop"])
        ax.set_title(row["label"], fontsize=10.5, weight="bold", color=ACCENT)
        ax.axis("off")
        for k in range(N_SIM):
            ax = fig.add_subplot(gs[r, 2 + k])
            ax.axis("off")
            if k < len(row["similar"]):
                ax.imshow(product_photo(row["similar"][k]))
            if k == 0:
                ax.set_title("most similar of 104K H&M products →", fontsize=10, weight="bold", loc="left")
        for k in range(N_REC):
            ax = fig.add_subplot(gs[r, 2 + N_SIM + 1 + k])
            ax.axis("off")
            if k < len(row["recs"]):
                ax.imshow(product_photo(row["recs"][k]))
            if k == 0:
                ax.set_title("buy next (LSTM) →", fontsize=10, weight="bold", loc="left")

    fig.suptitle("SnapShop: any clothing photo → tags, similar products, what to buy next",
                 fontsize=14, weight="bold", x=0.01, ha="left")
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--match", choices=list(ALPHA), default="balanced")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    drawn, rows = analyse(load_photo(args.image), device, args.match)
    out = ROOT / "reports" / "figures" / f"showcase_{args.image.stem}.png"
    draw(drawn, rows, out)
    print(f"saved {out}")
