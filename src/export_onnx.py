"""Export SnapShop's two CNNs to ONNX (fp32), check them against PyTorch, then make INT8 versions.
Run from the project root:  python src/export_onnx.py
"""
import gc
import json

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quant_pre_process, quantize_static)

from data.images import ProductImages, get_transforms
from predict import RUNS, load_models

OUT = RUNS / "onnx"
OUT.mkdir(exist_ok=True)


class Tagger(nn.Module):
    """The tagger as ONE forward pass: group logits, colour logits, colour/type fingerprint."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        feats = self.m.backbone(x)                                   # 2,048 numbers per image
        return self.m.group_head(feats), self.m.colour_head(feats), F.normalize(feats, dim=1)


def export(model, name, outputs):
    path = OUT / f"{name}.onnx"
    x = torch.randn(2, 3, 224, 224)                                  # example input (batch 2, see note)
    torch.onnx.export(model, (x,), path, dynamo=True, external_data=False,
                      input_names=["image"], output_names=outputs,
                      dynamic_shapes=({0: torch.export.Dim.DYNAMIC},))
    print(f"saved {path.name}: {path.stat().st_size / 1e6:.1f} MB")
    return path


def val_batch(n=32):
    """n real val images spread across the whole val set (already resized + normalised)."""
    ds = ProductImages("val")
    return torch.stack([ds[i][0] for i in range(0, len(ds), len(ds) // n)][:n])


@torch.no_grad()
def check(path, model, x):
    """Same input into PyTorch and ONNX Runtime -> largest difference per output."""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"image": x.numpy()})                  # None = all outputs
    torch_out = model(x)
    torch_out = torch_out if isinstance(torch_out, tuple) else (torch_out,)
    diffs = {}
    for out, a, b in zip(sess.get_outputs(), torch_out, onnx_out):
        a = a.numpy()
        line = f"  {path.name:14s} {out.name:14s} max |PyTorch - ONNX| = {np.abs(a - b).max():.1e}"
        if "logits" in out.name:                                      # same predicted class?
            line += f" | same class: {(a.argmax(1) == b.argmax(1)).mean() * 100:.0f}%"
        print(line)
        diffs[out.name] = float(np.abs(a - b).max())
    return diffs


class Calib(CalibrationDataReader):
    """Feeds 100 TRAIN images (eval transforms) so ONNX Runtime can measure each layer's value range.
    (Percentile calibration keeps every layer's activations for all images in RAM: 200 images crashed a 16 GB laptop.)"""
    def __init__(self, n=100, batch=4):
        ds = ProductImages("train")
        ds.tf = get_transforms(train=False)                           # no random crop/flip for calibration
        x = torch.stack([ds[i][0] for i in range(0, len(ds), len(ds) // n)][:n]).numpy()
        self.batches = iter([{"image": x[i:i + batch]} for i in range(0, len(x), batch)])

    def get_next(self):
        return next(self.batches, None)                               # None = no more data


def quantize(src):
    """fp32 .onnx -> INT8 .onnx (static quantization, QDQ format, per-channel weights)."""
    pre, dst = src.with_name(f"{src.stem}_pre.onnx"), src.with_name(f"{src.stem}_int8.onnx")
    # skip_symbolic_shape: ORT's symbolic shape tool can't handle the Expand op that F.normalize exports to;
    # ONNX's own shape inference + graph optimisation still run
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    # v1 (all op types, min/max ranges) lost 8-11 accuracy points and wrecked the fingerprints
    # (onnx_report: cosine 0.58). v2: quantize only the convolutions (~all the compute and weights),
    # keep the residual adds / pooling / heads / normalisation in fp32, and set each range at the
    # 99.999th percentile so a few extreme activations don't stretch it
    quantize_static(str(pre), str(dst), Calib(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv"], calibrate_method=CalibrationMethod.Percentile)
    pre.unlink()
    print(f"saved {dst.name}: {dst.stat().st_size / 1e6:.1f} MB")
    return dst


def run_onnx(path, x, batch=50):
    """All outputs of an .onnx model for a big array of images, in batches."""
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = [sess.run(None, {"image": x[i:i + batch]}) for i in range(0, len(x), batch)]
    return [np.concatenate(o) for o in zip(*outs)]


def compare_int8(t32, t8, d32, d8, n=1000):
    """fp32 vs INT8 on n val images: tag accuracy + how similar the fingerprints stay (cosine)."""
    ds = ProductImages("val")
    idx = list(range(0, len(ds), len(ds) // n))[:n]
    x = torch.stack([ds[i][0] for i in idx]).numpy()
    yg = np.array([ds.y_group[i] for i in idx])
    yc = np.array([ds.y_colour[i] for i in idx])
    rep, fps = {"n_val_images": len(idx)}, {}
    for name, path in [("fp32", t32), ("int8", t8)]:
        g, c, fps[name] = run_onnx(path, x)
        rep[f"{name}_group_acc"] = float((g.argmax(1) == yg).mean())
        rep[f"{name}_colour_acc"] = float((c.argmax(1) == yc).mean())
    d_fp32, d_int8 = run_onnx(d32, x)[0], run_onnx(d8, x)[0]
    for name, a, b in [("colour_fp", fps["fp32"], fps["int8"]), ("design_fp", d_fp32, d_int8)]:
        cos = (a * b).sum(1)                                          # both length 1 -> dot = cosine
        rep[f"{name}_cosine_fp32_vs_int8_mean"] = float(cos.mean())
        rep[f"{name}_cosine_fp32_vs_int8_min"] = float(cos.min())
    return rep


if __name__ == "__main__":
    tagger, designer = load_models("cpu")
    tagger = Tagger(tagger).eval()
    t_path = export(tagger, "tagger", ["group_logits", "colour_logits", "colour_fp"])
    d_path = export(designer, "designer", ["design_fp"])

    x = val_batch()
    print(f"\nPyTorch vs ONNX Runtime on {len(x)} val images:")
    report = {"pytorch_vs_onnx_max_diff": {**check(t_path, tagger, x), **check(d_path, designer, x)}}

    print("\nINT8 (static quantization of the convolutions, percentile calibration on 100 train images):")
    t8 = quantize(t_path)
    gc.collect()                                                      # free the first calibration's memory
    d8 = quantize(d_path)
    gc.collect()
    report["size_mb"] = {p.name: round(p.stat().st_size / 1e6, 1) for p in [t_path, t8, d_path, d8]}

    print("\nfp32 vs INT8 on 1,000 val images (takes a few minutes on CPU) ...")
    report["fp32_vs_int8"] = compare_int8(t_path, t8, d_path, d8)
    print(json.dumps(report, indent=2))
    (RUNS / "onnx_report.json").write_text(json.dumps(report, indent=2))
    print(f"saved {RUNS / 'onnx_report.json'}")
