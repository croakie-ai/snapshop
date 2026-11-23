"""SnapShop as a web API (FastAPI). Same models and functions as src/predict.py and the dashboard.

Endpoints
    GET  /health                        is the service up? device, catalogue size
    POST /predict                       upload a photo -> garments, tags, top-k similar, buy next (JSON)
    GET  /products/{article_id}/image   a catalogue product photo (so a front end can show results)
    GET  /docs                          interactive docs, generated automatically by FastAPI

Run from the project root:
    uvicorn api:app --app-dir src --port 8000
Try it:
    open http://127.0.0.1:8000/docs  ->  POST /predict  ->  "Try it out"  ->  upload a photo
    curl -F "image=@data/street_test/test1.jpg" "http://127.0.0.1:8000/predict?k=5"
"""
import io
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from pydantic import BaseModel
from ultralytics import YOLO

from data.images import IMG_DIR
from predict import (ALPHA, GROUP_OF, MARGIN, MAX_GARMENTS, MIN_CONF, SEARCH_POOL, YOLO_W, build_index,
                     buy_next, final_group, load_models, load_recommender, prepare, run_models, search, tags)

HIDE = {"Underwear", "Swimwear", "Nightwear"}
MAX_UPLOAD_MB = 10
M = {}                          # everything loaded once at startup (models, index, recommender)
LOCK = threading.Lock()         # one request at a time on the models / GPU


# ---------- response shapes (Pydantic): FastAPI validates them and documents them in /docs ----------
class Product(BaseModel):
    article_id: str
    group: str
    colour: str
    score: float | None = None


class Tags(BaseModel):
    group: str
    group_source: str           # tagger confidence, or "from detector" when YOLO's garment type wins
    colour: str
    colour_confidence: float


class Garment(BaseModel):
    label: str                  # e.g. "skirt" or "full photo (no garment detected)"
    detector_confidence: float | None
    box: list[float] | None     # x1, y1, x2, y2 in the photo's pixels
    tags: Tags
    similar: list[Product]
    buy_next_from: str | None   # the article id the recommender started from; None = trending (cold start)
    buy_next: list[Product]


class PredictResponse(BaseModel):
    match: str
    alpha: float
    n_garments: int
    time_ms: float
    garments: list[Garment]


# ---------- startup: load once, not per request ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    M["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    M["yolo"] = YOLO(str(YOLO_W))
    M["tagger"], M["designer"] = load_models(M["device"])
    M["alpha"] = ALPHA["balanced"]
    M["index"], M["meta"] = build_index(M["alpha"])
    M["info"] = {a: (g, c) for a, g, c in zip(M["meta"]["ids"], M["meta"]["group"], M["meta"]["colour"])}
    M["rec"] = load_recommender(M["device"])
    yield                       # the app serves requests here
    M.clear()                   # shutdown


app = FastAPI(title="SnapShop API", version="1.0",
              description="Clothing photo -> garments, tags, similar H&M products, buy next", lifespan=lifespan)


def get_index(alpha: float):
    """One search index in memory at a time (~1 GB); rebuilt only when a request asks for another match mode."""
    if M["alpha"] != alpha:
        M["index"] = None
        M["index"], M["meta"] = build_index(alpha)
        M["alpha"] = alpha
    return M["index"], M["meta"]


def find_garments(photo: Image.Image):
    """Same as predict.detect, but also keeps each box and confidence for the JSON."""
    r = M["yolo"].predict(photo, imgsz=640, conf=MIN_CONF, verbose=False)[0]
    found = []
    for (x1, y1, x2, y2), conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        mx, my = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
        box = (max(0, x1 - mx), max(0, y1 - my), min(photo.width, x2 + mx), min(photo.height, y2 + my))
        name = r.names[int(cls)]
        found.append((photo.crop(box), name, round(conf, 3), [round(v, 1) for v in (x1, y1, x2, y2)], GROUP_OF.get(name)))
    return found[:MAX_GARMENTS] or [(photo, "full photo (no garment detected)", None, None, None)]


def product(aid: str, score: float | None = None) -> Product:
    g, c = M["info"].get(aid, ("unknown", "unknown"))
    return Product(article_id=aid, group=g, colour=c, score=None if score is None else round(float(score), 4))


# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "device": M["device"], "catalogue_size": len(M["meta"]["ids"])}


@app.post("/predict", response_model=PredictResponse)
def predict(image: UploadFile = File(..., description="a clothing photo (jpg / png / webp)"),
            match: Literal["design", "balanced", "colour"] = "balanced",
            k: int = Query(10, ge=1, le=50, description="how many similar products"),
            detect: bool = True,
            hide_sensitive: bool = Query(True, description="leave underwear / swimwear / nightwear out")):
    data = image.file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"image larger than {MAX_UPLOAD_MB} MB")
    try:
        photo = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:           # UnidentifiedImageError, OSError, or whatever Ultralytics' patched
        raise HTTPException(400, "the upload is not a readable image")   # Image.open raises (it tries HEIF)

    t0 = time.perf_counter()
    alpha = ALPHA[match]
    with LOCK:                  # requests run in a thread pool; the models / GPU take one at a time
        index, meta = get_index(alpha)
        found = find_garments(photo) if detect else [(photo, "full photo", None, None, None)]
        garments = []
        for crop, label, conf, box, hint in found:
            out_g, out_c, colour_fp, design_fp = run_models(M["tagger"], M["designer"], prepare(crop), M["device"])
            t = tags(out_g, out_c, 1)
            group, why = final_group(t["group"], hint)
            scores, idx = search(index, design_fp, colour_fp, alpha, SEARCH_POOL, meta, hint)
            keep = [(s, j) for s, j in zip(scores, idx) if not (hide_sensitive and meta["group"][j] in HIDE)][:k]
            recs, seed = buy_next(M["rec"], [meta["ids"][j] for _, j in keep], M["device"], k=40)
            recs = [a for a in recs if not (hide_sensitive and M["info"].get(a, ("",))[0] in HIDE)][:12]
            garments.append(Garment(
                label=label, detector_confidence=conf, box=box,
                tags=Tags(group=group, group_source=why, colour=t["colour"][0][0],
                          colour_confidence=round(t["colour"][0][1], 3)),
                similar=[product(meta["ids"][j], s) for s, j in keep],
                buy_next_from=seed, buy_next=[product(a) for a in recs]))
    return PredictResponse(match=match, alpha=alpha, n_garments=len(garments),
                           time_ms=round((time.perf_counter() - t0) * 1000, 1), garments=garments)


@app.get("/products/{article_id}/image")
def product_image(article_id: str):
    if not (article_id.isdigit() and len(article_id) == 10):
        raise HTTPException(400, "article_id must be 10 digits, e.g. 0108775015")
    path = IMG_DIR / article_id[:3] / f"{article_id}.jpg"
    if not path.exists():
        raise HTTPException(404, "no photo for this article")
    return FileResponse(path, media_type="image/jpeg")
