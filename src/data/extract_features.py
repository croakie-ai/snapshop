"""Run every train/val photo through frozen ResNet-50 ONCE and save its 2,048-number fingerprint.

Why: in the frozen approach ResNet never changes, so a photo's fingerprint is identical every
time. Computing it once and saving it means training the new last layer later takes seconds,
not hours. Phase 4's visual search reuses these fingerprints too.

Uses : data/processed/splits.parquet + images_256/ (through images.py)
Makes: data/processed/resnet50_features.pt
       F_train / F_val     : fingerprints, shape (n, 2048), half precision
       G_train / G_val     : product-group answers (0-8)
       C_train / C_val     : colour answers (0-15)
       ids_train / ids_val : article_id of each row
Test photos are NOT processed (test stays untouched until the final evaluation).

Run from the project root:  python src/data/extract_features.py
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

from images import ProductImages, get_transforms

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "resnet50_features.pt"
BATCH_SIZE = 128          # lower to 64 if you get "CUDA out of memory"


def load_backbone(device: str) -> torch.nn.Module:
    """Pretrained ResNet-50 with its last layer removed, so it outputs 2,048 numbers per photo."""
    backbone = resnet50(weights=ResNet50_Weights.DEFAULT)   # ImageNet-trained weights (~100 MB download, once)
    backbone.fc = torch.nn.Identity()                        # remove the 1,000-category layer
    return backbone.to(device).eval()                        # eval = testing mode (frozen)


@torch.no_grad()                                             # no learning here, only computing
def extract(backbone, ds, device: str, desc: str):
    """Pass every photo of a dataset through ResNet; return fingerprints + group + colour answers."""
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    feats, groups, colours = [], [], []
    for x, g, c in tqdm(dl, desc=desc):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            f = backbone(x.to(device, non_blocking=True))   # (batch, 2048)
        feats.append(f.float().cpu())
        groups.append(g)
        colours.append(c)
    return torch.cat(feats), torch.cat(groups), torch.cat(colours)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = load_backbone(device)
    print(f"Device: {device} | ResNet-50 parameters: {sum(p.numel() for p in backbone.parameters()):,}")

    out = {}
    for split in ["train", "val"]:
        ds = ProductImages(split)
        ds.tf = get_transforms(train=False)      # centre crop, no randomness: ONE fixed fingerprint per photo
        F, G, C = extract(backbone, ds, device, desc=split)
        out[f"F_{split}"] = F.half()             # half precision: halves the file size
        out[f"G_{split}"] = G
        out[f"C_{split}"] = C
        out[f"ids_{split}"] = ds.ids
        print(f"{split}: fingerprints {tuple(F.shape)}")

    torch.save(out, OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()