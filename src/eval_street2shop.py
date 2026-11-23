"""Phase 5.4 + 5.5: street-to-shop search on DeepFashion2, crop ablation, and the final alpha.

The real SnapShop task: a STREET photo of a garment (DF2 source = "user") -> search a gallery of
SHOP photos -> is the SAME item (same pair_id + same style id) in the top K?  Unlike the H&M sibling
test, the target here has the same design AND the same colour, so there is no built-in conflict
between the two halves of the hybrid -> this test chooses the final alpha (D24).

Query variants (5.4 crop ablation); the gallery is fixed (all shop items, ground-truth box crop):
  full : the whole street photo (no detector)
  box  : ground-truth box crop                       (a perfect detector)
  mask : ground-truth outline, background -> white   (a perfect detector + segmentation)
  yolo : OUR YOLO11s-seg: its detection that best overlaps the item, outline on white;
         full photo if YOLO finds nothing there (IoU < 0.3)  = the real SnapShop pipeline
Search (5.5): metric (design) + ft_ls (colour/type) fingerprints, alpha 0 / 0.25 / 0.5 / 0.75 / 1.
None of the search models ever saw DeepFashion2; YOLO never trained on DF2 validation images.

Makes: runs/street2shop_results.csv  (+ printed tables)
Run from the project root:  python src/eval_street2shop.py
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ultralytics import YOLO

from data.images import get_transforms
from data.make_images import SIZE, pad_to_square
from predict import load_models

ROOT = Path(__file__).resolve().parents[1]
DF2 = ROOT / "data" / "raw" / "deepfashion2" / "validation"
YOLO_W = ROOT / "runs" / "segment" / "runs" / "yolo" / "df2_y11s" / "weights" / "best.pt"
N_QUERIES, SEED = 3_000, 42
MODES = ["full", "box", "mask", "yolo"]
ALPHAS = [0, 0.25, 0.5, 0.75, 1]
KS = [1, 5, 10, 20]
TF = get_transforms(train=False)


def load_items():
    """Every garment in DF2 validation with a style id > 0 (only those can be matched across photos)."""
    items = []
    for p in tqdm(sorted((DF2 / "annos").glob("*.json")), desc="reading DF2 labels"):
        a = json.loads(p.read_text())
        for k, it in a.items():
            if k.startswith("item") and it["style"] > 0:
                items.append({"img": p.stem, "source": a["source"], "key": (a["pair_id"], it["style"]),
                              "box": it["bounding_box"], "seg": it["segmentation"]})
    return items


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def crop(img, box, polys, mode):
    out = _crop(img, box, polys, mode)
    return out if min(out.size) >= 8 else img          # guard against empty/tiny crops


def _crop(img, box, polys, mode):
    if mode == "full" or (mode == "yolo" and polys is None):
        return img
    if mode == "box":
        return img.crop(tuple(box))
    m = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(m)
    for p in polys:
        if len(p) >= 3:
            draw.polygon([tuple(pt) for pt in p], fill=255)
    out = Image.composite(img, Image.new("RGB", img.size, (255, 255, 255)), m)
    return out.crop(m.getbbox() or tuple(box))


class Crops(Dataset):
    def __init__(self, specs):
        self.specs = specs                     # (image id, box, polygons or None, mode)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, i):
        img_id, box, polys, mode = self.specs[i]
        img = Image.open(DF2 / "image" / f"{img_id}.jpg").convert("RGB")
        c = crop(img, box, polys, mode)
        return TF(pad_to_square(c).resize((SIZE, SIZE), Image.Resampling.LANCZOS))


@torch.no_grad()
def embed(specs, tagger, designer, device):
    D, C = [], []
    for x in tqdm(DataLoader(Crops(specs), batch_size=64, num_workers=2), desc="fingerprints"):
        x = x.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            feats, d = tagger.backbone(x), designer(x)
        C.append(F.normalize(feats.float(), dim=1).cpu())
        D.append(F.normalize(d.float(), dim=1).cpu())
    return torch.cat(D), torch.cat(C)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = load_items()
    shop = [it for it in items if it["source"] == "shop"]
    shop_keys = {it["key"] for it in shop}
    user = [it for it in items if it["source"] == "user" and it["key"] in shop_keys]
    queries = random.Random(SEED).sample(user, min(N_QUERIES, len(user)))
    print(f"gallery: {len(shop):,} shop items | queries: {len(queries):,} street items (of {len(user):,} with a match)")

    # 1) our YOLO on the street photos: keep the detection that best overlaps each queried item
    yolo, yolo_polys, yolo_boxes = YOLO(str(YOLO_W)), [], []
    for q in tqdm(queries, desc="YOLO"):
        r = yolo.predict(str(DF2 / "image" / f"{q['img']}.jpg"), imgsz=640, conf=0.25, verbose=False)[0]
        best, best_box, best_iou = None, None, 0.3
        if r.masks is not None:
            for b, poly in zip(r.boxes.xyxy.cpu().numpy(), r.masks.xy):
                v = iou(b, q["box"])
                if v > best_iou:
                    best, best_box, best_iou = poly, [int(c) for c in b], v
        yolo_polys.append([best] if best is not None else None)
        yolo_boxes.append(best_box)          # YOLO's own box: used if its mask comes out empty
    found = sum(p is not None for p in yolo_polys)
    print(f"YOLO found the queried garment in {found:,} / {len(queries):,} street photos ({found / len(queries):.1%})")
    del yolo
    torch.cuda.empty_cache()

    # 2) fingerprints: gallery once, queries once per crop mode
    tagger, designer = load_models(device)
    gDs, gCs = embed([(it["img"], it["box"], None, "box") for it in shop], tagger, designer, device)
    key_id = {k: i for i, k in enumerate(sorted({it["key"] for it in items}))}
    g_keys = torch.tensor([key_id[it["key"]] for it in shop])
    q_keys = torch.tensor([key_id[q["key"]] for q in queries])
    gt_polys = [[np.array(p).reshape(-1, 2) for p in q["seg"] if len(p) >= 6] for q in queries]

    rows = []
    for mode in MODES:
        if mode == "yolo":                   # the real pipeline: only YOLO's own outline / box
            specs = [(q["img"], yb or q["box"], pl, mode) for q, pl, yb in zip(queries, yolo_polys, yolo_boxes)]
        else:                                # ground truth: a perfect detector
            specs = [(q["img"], q["box"], pl, mode) for q, pl in zip(queries, gt_polys)]
        qD, qC = embed(specs, tagger, designer, device)
        for a in ALPHAS:
            top = (a * qD @ gDs.T + (1 - a) * qC @ gCs.T).topk(max(KS), dim=1).indices
            hit = g_keys[top] == q_keys[:, None]
            rows.append({"query crop": mode, "alpha": a,
                         **{f"Recall@{k}": hit[:, :k].any(1).float().mean().item() for k in KS}})

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "runs" / "street2shop_results.csv", index=False)
    print("\nRecall@10  (rows = how the street photo is cropped, columns = alpha: 0 colour/type ... 1 design)")
    print(df.pivot(index="query crop", columns="alpha", values="Recall@10").reindex(MODES).round(3).to_string())
    print("\nRecall@1")
    print(df.pivot(index="query crop", columns="alpha", values="Recall@1").reindex(MODES).round(3).to_string())
    print(f"\nSaved: {ROOT / 'runs' / 'street2shop_results.csv'}")


if __name__ == "__main__":
    main()
