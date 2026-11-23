"""SnapShop dashboard: drop in any photo -> garments (YOLO) -> tags + top-k similar H&M products + buy next.

It reuses the SAME functions as src/predict.py (detection settings, tagger, hybrid search, recommender),
so the dashboard and the command line always give the same answers. Everything runs locally (no internet).

Run from the project root (conda env "snapshop"):
    pip install gradio          (once)
    python src/app.py           -> opens http://127.0.0.1:7860 in the browser
"""
import json
import os
import time

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"   # offline: no usage pings

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO

from data.images import IMG_DIR
from predict import (ALPHA, GROUP_OF, MARGIN, MAX_GARMENTS, MIN_CONF, ROOT, RUNS, SEARCH_POOL, YOLO_W,
                     build_index, buy_next, final_group, load_models, load_photo, load_recommender,
                     prepare, run_models, search, tags)

HIDE = {"Underwear", "Swimwear", "Nightwear"}       # "presentation mode": keep these out of the results
N_RECS = 12
DEMO_DIR = ROOT / "data" / "street_test"
YOLO_TEST = RUNS / "segment" / "runs" / "yolo" / "df2_test"
FIG = ROOT / "reports" / "figures"

# ---------- load everything once (takes ~30 s) ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"loading models on {DEVICE} ...")
DETECTOR = YOLO(str(YOLO_W))
TAGGER, DESIGNER = load_models(DEVICE)
REC = load_recommender(DEVICE)
_cache = {"alpha": None, "index": None, "meta": None}


def get_index(alpha: float):
    """One search index in memory at a time (each is ~1 GB); rebuilt only when the match mode changes."""
    if _cache["alpha"] != alpha:
        _cache["index"] = None
        _cache["index"], _cache["meta"] = build_index(alpha)
        _cache["alpha"] = alpha
    return _cache["index"], _cache["meta"]


_, META = get_index(ALPHA["balanced"])
INFO = {a: (g, c) for a, g, c in zip(META["ids"], META["group"], META["colour"])}   # article id -> (group, colour)
print("ready")


def img_path(aid: str) -> str:
    return str(IMG_DIR / aid[:3] / f"{aid}.jpg")


def short(group: str) -> str:
    return group.replace("Garment ", "").replace(" body", "")     # "Garment Upper body" -> "Upper"


def detect_and_draw(photo: Image.Image):
    """Same as predict.detect, but also returns the photo with YOLO's boxes + outlines drawn on it."""
    r = DETECTOR.predict(photo, imgsz=640, conf=MIN_CONF, verbose=False)[0]
    drawn = Image.fromarray(np.ascontiguousarray(r.plot()[..., ::-1]))     # BGR -> RGB
    crops = []
    for (x1, y1, x2, y2), conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        mx, my = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
        box = (max(0, x1 - mx), max(0, y1 - my), min(photo.width, x2 + mx), min(photo.height, y2 + my))
        name = r.names[int(cls)]
        crops.append((photo.crop(box), f"{name} ({conf:.2f})", GROUP_OF.get(name)))
    return crops[:MAX_GARMENTS] or [(photo, "full photo (no garment detected)", None)], drawn


def analyse(path, match, k, use_yolo, use_hint, safe):
    if not path:
        raise gr.Error("Upload a photo (or click an example) first.")
    t0 = time.time()
    alpha = ALPHA[match]
    index, meta = get_index(alpha)
    photo = load_photo(path)
    if use_yolo:
        garments, drawn = detect_and_draw(photo)
    else:
        garments, drawn = [(photo, "full photo (detector off)", None)], photo

    outs = []
    for n in range(MAX_GARMENTS):
        if n >= len(garments):
            outs += [gr.update(visible=False), None, "", [], []]
            continue
        crop, label, hint = garments[n]
        hint = hint if use_hint else None
        out_g, out_c, colour_fp, design_fp = run_models(TAGGER, DESIGNER, prepare(crop), DEVICE)
        t = tags(out_g, out_c, 3)
        group, why = final_group(t["group"], hint)

        # search a pool of 200, then (optionally) drop underwear/swimwear/nightwear and keep the top k
        scores, idx = search(index, design_fp, colour_fp, alpha, SEARCH_POOL, meta, hint)
        keep = [(s, j) for s, j in zip(scores, idx) if not (safe and meta["group"][j] in HIDE)][:k]
        similar = [(img_path(meta["ids"][j]), f"#{r} {short(meta['group'][j])} · {meta['colour'][j]} ({s:.2f})")
                   for r, (s, j) in enumerate(keep, start=1)]

        recs, seed = buy_next(REC, [meta["ids"][j] for _, j in keep], DEVICE, k=40)
        recs = [a for a in recs if a in INFO and not (safe and INFO[a][0] in HIDE)][:N_RECS]
        buy = [(img_path(a), f"{short(INFO[a][0])} · {INFO[a][1]}") for a in recs]

        g_list = ", ".join(f"{short(nm)} {c * 100:.0f}%" for nm, c in t["group"])
        c_list = ", ".join(f"{nm} {c * 100:.0f}%" for nm, c in t["colour"])
        src = f"after `{seed}` (the first match the LSTM knows)" if seed else "trending this week (cold start)"
        md = (f"### Garment {n + 1}: {label}\n"
              f"**Type:** {short(group)} ({why}) &nbsp;&nbsp; **Colour:** {t['colour'][0][0]} "
              f"({t['colour'][0][1] * 100:.0f}%)\n\n"
              f"<small>tagger group guesses: {g_list}<br>colour guesses: {c_list}<br>"
              f"buy next: {REC['kind']}, {src}</small>")
        outs += [gr.update(visible=True), crop, md, similar, buy]

    summary = (f"**{len(garments)} garment(s)** · match = **{match}** (α = {alpha}) · "
               f"{time.time() - t0:.1f} s on {DEVICE}")
    return [drawn, summary] + outs


# ---------- results tab ----------
def results_md() -> str:
    md = ["## Validation (used for every decision)",
          "| Component | Result |", "|---|---|",
          "| Tags: majority → MLP → frozen → fine-tuned → **+ label smoothing** | group 40.7 → 83.4 → 89.3 → 93.7 → **94.1%**; colour 21.2 → 71.6 → 70.6 → 82.0 → **82.3%** |",
          "| Search Recall@10 (design siblings vs 104k) | ImageNet 0.360 · fine-tuned 0.181 · **ArcFace 0.799** · hybrid α 0.5: 0.491 |",
          "| Street-to-shop Recall@10 (α 0.5) | full photo 0.125 · **YOLO crop 0.204** · perfect box 0.271 · perfect mask 0.266 |",
          "| Buy next MAP@12 (full / last item) | popularity 0.0066 · co-purchase 0.0115 / 0.0097 · **LSTM 0.0217 / 0.0197** · transformer 0.0215 / 0.0198 |",
          ""]
    p = RUNS / "test_results.json"
    if p.exists():
        r = json.loads(p.read_text())
        c = r["classifier_ft_ls"]
        md += ["## Test (held out, run once)", "| Component | Result |", "|---|---|",
               f"| Tags ({c['n']:,} products) | group **{c['group_acc'] * 100:.1f}%** (macro-F1 {c['group_macro_f1']:.3f}) · "
               f"colour **{c['colour_acc'] * 100:.1f}%** (macro-F1 {c['colour_macro_f1']:.3f}) |"]
        for name, s in r["search"].items():
            if isinstance(s, dict):
                md.append(f"| Search: {name} | R@1 {s['recall@1']:.3f} · R@5 {s['recall@5']:.3f} · "
                          f"**R@10 {s['recall@10']:.3f}** · same type {s['group_match@10'] * 100:.0f}% · "
                          f"same colour {s['colour_match@10'] * 100:.0f}% |")
        for name, s in r["recommender_map12"].items():
            if isinstance(s, dict):
                md.append(f"| MAP@12: {name} | full {s['full']:.4f} · last item {s['last_item']:.4f} |")
        md.append("| Detector (5,000 unseen DF2 photos) | box mAP50 **0.779** · mask mAP50 0.600 |")
    return "\n".join(md)


def figures() -> list:
    files = [(FIG / "snapshop_architecture_final.png", "Final architecture"),
             (FIG / "snapshop_phases.png", "Phases"),
             (FIG / "split_explained.png", "Leakage-free split"),
             (FIG / "mlp_curves.png", "MLP curves"),
             (FIG / "finetune_curves.png", "Fine-tune curves"),
             (FIG / "finetune_confusion.png", "Confusion matrices"),
             (YOLO_TEST / "BoxPR_curve.png", "YOLO box PR curve"),
             (YOLO_TEST / "MaskPR_curve.png", "YOLO mask PR curve"),
             (FIG / "weekly_purchases.png", "Weekly purchases")]
    return [(str(p), cap) for p, cap in files if p.exists()]


def street2shop():
    p = RUNS / "street2shop_results.csv"
    return pd.read_csv(p).round(3) if p.exists() else None


EXAMPLES = sorted((DEMO_DIR / "demo").glob("*.jpg")) + sorted(DEMO_DIR.glob("person_*.jpg"))

with gr.Blocks(title="SnapShop") as app:
    gr.Markdown("# 🛍️ SnapShop — street-to-shop visual search\n"
                "Any clothing photo → **YOLO** finds each garment → **tags** (type + colour) → "
                "**top-k similar** H&M products (104,014) → **buy next** (LSTM on 31M purchases)")
    with gr.Tab("Try a photo"):
        with gr.Row():
            with gr.Column(scale=1):
                photo_in = gr.Image(type="filepath", label="Photo", height=360)
                match = gr.Radio(list(ALPHA), value="balanced", label="Match by",
                                 info="design = same style (α 1) · balanced (α 0.5, final) · colour (α 0.25)")
                k = gr.Slider(5, 20, value=10, step=1, label="How many similar products")
                with gr.Row():
                    use_yolo = gr.Checkbox(True, label="Detector (YOLO)")
                    use_hint = gr.Checkbox(True, label="YOLO type hint")
                    safe = gr.Checkbox(True, label="Hide underwear / swim / nightwear")
                go = gr.Button("Search", variant="primary")
                gr.Examples([[str(p)] for p in EXAMPLES], inputs=[photo_in], label="Demo photos")
            with gr.Column(scale=1):
                drawn = gr.Image(label="What YOLO found", height=360)
                summary = gr.Markdown()
        garment_outputs = []
        for n in range(MAX_GARMENTS):
            with gr.Column(visible=False) as col:
                with gr.Row():
                    crop = gr.Image(label=f"Garment {n + 1}", height=220, scale=1)
                    info = gr.Markdown()
                sim = gr.Gallery(label="Most similar H&M products", columns=5, height=540, object_fit="contain")
                rec = gr.Gallery(label="Buy next (LSTM)", columns=6, height=460, object_fit="contain")
            garment_outputs += [col, crop, info, sim, rec]
        go.click(analyse, [photo_in, match, k, use_yolo, use_hint, safe], [drawn, summary] + garment_outputs)

    with gr.Tab("Results"):
        gr.Markdown(results_md())
        gr.Markdown("### Street-to-shop crop ablation (3,000 DeepFashion2 street photos)")
        gr.Dataframe(street2shop())
        gr.Gallery(figures(), label="Figures", columns=3, height=600, object_fit="contain")

    with gr.Tab("How it works"):
        gr.Markdown(
            "1. **YOLO11s-seg** (trained on 30k DeepFashion2 photos) finds up to 3 garments (conf ≥ 0.5), box crop + 5% margin.\n"
            "2. Each crop: white-pad to square → 256 → centre-crop 224 → ImageNet normalise (same as training).\n"
            "3. **Fine-tuned ResNet-50 + label smoothing** → group + colour tags and a 2,048-d colour/type fingerprint. "
            "If it disagrees with YOLO's garment type (or is < 60% sure), YOLO's type wins.\n"
            "4. **ResNet-50 + ArcFace** → 512-d design fingerprint.\n"
            "5. **Hybrid search**: score = α·cos(design) + (1−α)·cos(colour/type) over all 104,014 products (FAISS, exact). "
            "Same-type products first.\n"
            "6. **LSTM** next-item model: the first match it knows → 12 most likely next purchases; "
            "unknown → this week's best-sellers.\n\n"
            "Code: `src/predict.py` (pipeline), `src/app.py` (this dashboard). Decisions: `reports/decisions.md`.")

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", inbrowser=True, allowed_paths=[str(IMG_DIR), str(ROOT)])
