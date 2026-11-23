"""Turn every catalogue product photo into a fingerprint (embedding) for visual search (Phase 4).

Backbones (pick with --model):
  imagenet : plain ImageNet ResNet-50, never trained on our data   (baseline)
  ft       : our fine-tuned ResNet-50, run A                       (Phase 3)
  ft_ls    : fine-tuned + label smoothing, run B                   (Phase 3 best)

Each photo -> 2,048 numbers -> scaled to length 1 (L2-normalised), so that
cosine similarity = a simple dot product (what FAISS will compute).

Uses : splits.parquet + images_256/ (ALL 104k products: the whole catalogue is searchable)
Makes: data/processed/emb_<model>.pt  {emb (N, 2048), ids, product_code, split, group, colour}

Run from the project root:  python src/extract_embeddings.py --model imagenet
"""
import argparse
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

from data.images import IMG_DIR, SPLITS, get_transforms
from train_finetune import MultiHeadResNet

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
WEIGHTS = {"ft": ROOT / "runs" / "finetuned_resnet50.pt",
           "ft_ls": ROOT / "runs" / "finetuned_resnet50_ls.pt"}


class Catalogue(Dataset):
    """Every product photo, with the val/test transform (no random crop or flip)."""

    def __init__(self, df: pl.DataFrame):
        self.ids = df["article_id"].to_list()
        self.tf = get_transforms(train=False)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        aid = self.ids[i]
        return self.tf(Image.open(IMG_DIR / aid[:3] / f"{aid}.jpg").convert("RGB"))


def load_backbone(name: str) -> torch.nn.Module:
    if name == "imagenet":
        net = resnet50(weights=ResNet50_Weights.DEFAULT)
        net.fc = torch.nn.Identity()
        return net
    model = MultiHeadResNet()
    model.load_state_dict(torch.load(WEIGHTS[name], map_location="cpu"))
    return model.backbone            # drop the two tag heads, keep the 2,048-number fingerprint


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["imagenet", "ft", "ft_ls"], required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pl.read_parquet(SPLITS)
    net = load_backbone(args.model).to(device).eval()
    dl = DataLoader(Catalogue(df), batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    chunks = []
    for x in tqdm(dl, desc=f"embedding ({args.model})"):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            f = net(x.to(device, non_blocking=True))
        chunks.append(F.normalize(f.float(), dim=1).half().cpu())   # length 1 -> cosine = dot product

    emb = torch.cat(chunks)
    out = OUT / f"emb_{args.model}.pt"
    torch.save({"emb": emb, "ids": df["article_id"].to_list(), "product_code": df["product_code"].to_list(),
                "split": df["split"].to_list(), "group": df["group"].to_list(), "colour": df["colour"].to_list()}, out)
    print(f"Saved {tuple(emb.shape)} -> {out}")


if __name__ == "__main__":
    main()