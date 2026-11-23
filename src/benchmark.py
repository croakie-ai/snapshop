"""Latency benchmark: how long SnapShop takes per photo.

Part 1, the two CNNs (tagger + ArcFace designer) on one garment crop, in 4 setups:
    PyTorch GPU | PyTorch CPU | ONNX Runtime CPU fp32 | ONNX Runtime CPU INT8
Part 2, the full pipeline on real photos (GPU): YOLO -> tags + fingerprints -> search 104k -> buy next

Batch size 1 (one user's photo). Warm-up runs first (not timed), then timed runs -> p50 / p95 in ms.
Run from the project root, after export_onnx.py:   python src/benchmark.py
"""
import json
import platform
import time

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

from data.images import ProductImages
from export_onnx import OUT, Tagger
from predict import (ALPHA, ROOT, RUNS, YOLO_W, build_index, buy_next, detect, load_models, load_photo,
                     load_recommender, prepare, run_models, search)


def timeit(fn, inputs, warmup=10, runs=100):
    """Call fn on inputs (cycling through them); return p50 / p95 of the timed runs in milliseconds."""
    for i in range(warmup):
        fn(inputs[i % len(inputs)])
    ms = []
    for i in range(runs):
        t0 = time.perf_counter()
        fn(inputs[i % len(inputs)])
        ms.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": round(float(np.percentile(ms, 50)), 1), "p95_ms": round(float(np.percentile(ms, 95)), 1)}


def torch_fn(tagger, designer, device):
    @torch.no_grad()
    def f(x):
        x = x.to(device)
        tagger(x)
        designer(x)
        if device == "cuda":
            torch.cuda.synchronize()          # GPU work is asynchronous: wait for it to really finish
    return f


def onnx_fn(t_path, d_path):
    t = ort.InferenceSession(str(t_path), providers=["CPUExecutionProvider"])
    d = ort.InferenceSession(str(d_path), providers=["CPUExecutionProvider"])

    def f(x):
        a = x.numpy()
        t.run(None, {"image": a})
        d.run(None, {"image": a})
    return f


def pipeline_fn(device):
    """Everything the app does for one photo (index + models loaded once, like a server)."""
    yolo = YOLO(str(YOLO_W))
    tagger, designer = load_models(device)
    index, meta = build_index(ALPHA["balanced"])
    rec = load_recommender(device)

    def f(path):
        photo = load_photo(path)
        for crop, _, hint in detect(photo, yolo):
            _, _, colour_fp, design_fp = run_models(tagger, designer, prepare(crop), device)
            _, idx = search(index, design_fp, colour_fp, ALPHA["balanced"], 10, meta, hint)
            buy_next(rec, [meta["ids"][j] for j in idx], device)
    return f


if __name__ == "__main__":
    ds = ProductImages("val")
    crops = [ds[i][0].unsqueeze(0) for i in range(0, len(ds), len(ds) // 100)][:100]   # 100 single images
    result = {"hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                           "cpu": platform.processor()},
              "models_per_crop": {}}

    tagger, designer = load_models("cpu")
    tagger = Tagger(tagger).eval()
    setups = []
    if torch.cuda.is_available():
        setups.append(("PyTorch GPU (fp32)", lambda: torch_fn(tagger.cuda(), designer.cuda(), "cuda")))
    setups.append(("PyTorch CPU (fp32)", lambda: torch_fn(tagger.cpu(), designer.cpu(), "cpu")))
    setups.append(("ONNX Runtime CPU (fp32)", lambda: onnx_fn(OUT / "tagger.onnx", OUT / "designer.onnx")))
    if (OUT / "tagger_int8.onnx").exists():
        setups.append(("ONNX Runtime CPU (INT8)", lambda: onnx_fn(OUT / "tagger_int8.onnx", OUT / "designer_int8.onnx")))

    print("Tagger + designer on one crop (batch 1, 10 warm-up + 100 timed runs):")
    for name, make in setups:
        r = timeit(make(), crops)
        result["models_per_crop"][name] = r
        print(f"  {name:26s} p50 {r['p50_ms']:7.1f} ms   p95 {r['p95_ms']:7.1f} ms")
    del tagger, designer
    torch.cuda.empty_cache()

    photos = sorted((ROOT / "data" / "street_test" / "demo").glob("*.jpg"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nFull pipeline on {len(photos)} demo photos ({device}; 5 warm-up + 40 timed runs) ...")
    r = timeit(pipeline_fn(device), photos, warmup=5, runs=40)
    result["full_pipeline_per_photo"] = {"device": device, "n_photos": len(photos), **r}
    print(f"  full pipeline              p50 {r['p50_ms']:7.1f} ms   p95 {r['p95_ms']:7.1f} ms")

    (RUNS / "benchmark.json").write_text(json.dumps(result, indent=2))
    print(f"\nsaved {RUNS / 'benchmark.json'}")
