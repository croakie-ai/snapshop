"""Run the full SnapShop pipeline on some photos and save the raw results as JSON (for making figures).

For each photo: YOLO boxes, each garment's tags, its top-10 similar products and 12 buy-next picks
(underwear / swimwear / nightwear left out). Saves reports/demo_results.json.

Run from the project root:   python src/export_demo.py data/street_test/person_*.jpg data/street_test/demo/*.jpg
"""
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from predict import (ALPHA, GROUP_OF, MARGIN, MAX_GARMENTS, MIN_CONF, ROOT, SEARCH_POOL, YOLO_W, build_index,
                     buy_next, final_group, load_models, load_photo, load_recommender, prepare, run_models,
                     search, tags)

HIDE = {"Underwear", "Swimwear", "Nightwear"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("images", type=Path, nargs="+")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(str(YOLO_W))
    tagger, designer = load_models(device)
    alpha = ALPHA["balanced"]
    index, meta = build_index(alpha)
    info = {a: (g, c) for a, g, c in zip(meta["ids"], meta["group"], meta["colour"])}
    rec = load_recommender(device)

    out = []
    for path in args.images:
        photo = load_photo(path)
        r = yolo.predict(photo, imgsz=640, conf=MIN_CONF, verbose=False)[0]
        garments = []
        for (x1, y1, x2, y2), conf, cls in list(zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()))[:MAX_GARMENTS]:
            mx, my = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
            box = (max(0, x1 - mx), max(0, y1 - my), min(photo.width, x2 + mx), min(photo.height, y2 + my))
            name = r.names[int(cls)]
            hint = GROUP_OF.get(name)
            out_g, out_c, colour_fp, design_fp = run_models(tagger, designer, prepare(photo.crop(box)), device)
            t = tags(out_g, out_c, 3)
            group, why = final_group(t["group"], hint)
            scores, idx = search(index, design_fp, colour_fp, alpha, SEARCH_POOL, meta, hint)
            keep = [(float(s), meta["ids"][j]) for s, j in zip(scores, idx) if meta["group"][j] not in HIDE][:10]
            recs, seed = buy_next(rec, [a for _, a in keep], device, k=40)
            recs = [a for a in recs if a in info and info[a][0] not in HIDE][:12]
            garments.append({
                "yolo_class": name, "yolo_conf": round(conf, 3), "box": [round(v, 1) for v in (x1, y1, x2, y2)],
                "crop_box": [round(v, 1) for v in box],
                "group": group, "group_source": why, "group_top3": t["group"],
                "colour": t["colour"][0][0], "colour_conf": round(t["colour"][0][1], 3), "colour_top3": t["colour"],
                "similar": [{"id": a, "score": round(s, 4), "group": info[a][0], "colour": info[a][1]} for s, a in keep],
                "buy_next_seed": seed,
                "buy_next": [{"id": a, "group": info[a][0], "colour": info[a][1]} for a in recs]})
        out.append({"photo": str(path).replace("\\", "/"), "width": photo.width, "height": photo.height,
                    "garments": garments})
        print(f"{path.name}: {len(garments)} garment(s): " + ", ".join(f"{g['yolo_class']} -> {g['group']} / {g['colour']}" for g in garments))

    dst = ROOT / "reports" / "demo_results.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"saved {dst}")
