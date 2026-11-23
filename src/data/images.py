"""PyTorch image dataset for the CNN phases (Phase 3 onwards).

Reads the cached 256x256 product photos (data/processed/images_256/) for one split of
splits.parquet, and returns (image tensor, group index, colour index) for each product.

Class indices are alphabetical, exactly as in the Phase 2 MLP dataset, so results compare 1:1.
"""
from pathlib import Path

import polars as pl
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "processed" / "splits.parquet"
IMG_DIR = ROOT / "data" / "processed" / "images_256"

# ResNet was pretrained on ImageNet photos standardized with these per-channel numbers,
# so our images must be standardized the same way (same idea as Phase 2's standardization).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool) -> transforms.Compose:
    """Train: random 224 crop + random left-right flip (augmentation). Val/test: fixed centre crop.

    No colour augmentation (brightness/hue changes): colour is one of our targets, so changing
    it would make the labels wrong.
    """
    if train:
        steps = [transforms.RandomCrop(224), transforms.RandomHorizontalFlip()]
    else:
        steps = [transforms.CenterCrop(224)]
    return transforms.Compose(steps + [
        transforms.ToTensor(),                              # 0-255 pixels -> 0-1, shape (3, 224, 224)
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # standardize like ImageNet
    ])


def class_lists() -> tuple[list[str], list[str]]:
    df = pl.read_parquet(SPLITS)
    return sorted(df["group"].unique().to_list()), sorted(df["colour"].unique().to_list())


class ProductImages(Dataset):
    """One split ("train" / "val" / "test") of the product photos."""

    def __init__(self, split: str):
        df = pl.read_parquet(SPLITS).filter(pl.col("split") == split)
        groups, colours = class_lists()
        g_idx = {name: i for i, name in enumerate(groups)}
        c_idx = {name: i for i, name in enumerate(colours)}

        self.ids = df["article_id"].to_list()
        self.y_group = [g_idx[g] for g in df["group"].to_list()]
        self.y_colour = [c_idx[c] for c in df["colour"].to_list()]
        self.tf = get_transforms(train=(split == "train"))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        aid = self.ids[i]
        img = Image.open(IMG_DIR / aid[:3] / f"{aid}.jpg").convert("RGB")
        return self.tf(img), self.y_group[i], self.y_colour[i]