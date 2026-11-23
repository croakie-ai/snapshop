"""Metric learning for visual search (Phase 4): ResNet-50 + ArcFace, trained so that
photos of the SAME DESIGN (same product_code, any colour) get close fingerprints.

- Start: ImageNet ResNet-50 (best design baseline, D23), fc removed -> 2,048 features
- Embedding head: Linear 2048 -> 512 + BatchNorm -> L2-normalised 512-number fingerprint
- ArcFace head (training only): one learned "centre" per train design; the loss pulls each
  fingerprint toward its design's centre, with an extra angle margin so designs stay well apart
- Colour jitter ON: this is the DESIGN half of the hybrid search; colour comes from the
  fine-tuned classifier, so here we teach the model to ignore colour
- Model selection: val Recall@10 (siblings searched among val products) after every epoch

Makes: runs/metric_arcface.pt (best epoch), runs/metric_history.json
Run from the project root:  python src/train_metric.py
"""
import json
import time
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

from data.images import IMG_DIR, IMAGENET_MEAN, IMAGENET_STD, SPLITS, get_transforms

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

EPOCHS, PATIENCE, BATCH_SIZE = 10, 3, 64
LR_BACKBONE, LR_HEADS = 1e-4, 1e-3
EMB_DIM, SCALE, MARGIN = 512, 30.0, 0.3
SEED = 42

TRAIN_TF = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4),   # cheap colour changes
    transforms.RandomGrayscale(p=0.2),                                       # 20%: no colour at all -> colour != design
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class Designs(Dataset):
    def __init__(self, df: pl.DataFrame, tf):
        self.ids = df["article_id"].to_list()
        self.labels = df["label"].to_list()
        self.tf = tf

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        aid = self.ids[i]
        img = Image.open(IMG_DIR / aid[:3] / f"{aid}.jpg").convert("RGB")
        return self.tf(img), self.labels[i]


class Embedder(nn.Module):
    """Photo -> 512-number fingerprint of length 1."""

    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(nn.Linear(2048, EMB_DIM), nn.BatchNorm1d(EMB_DIM))

    def forward(self, x):
        return F.normalize(self.head(self.backbone(x)), dim=1)


class ArcFace(nn.Module):
    """Score for design j = s*cos(angle_j); for the TRUE design: s*cos(angle + m)."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, EMB_DIM) * 0.01)   # one centre per design

    def forward(self, emb, y):
        cos = emb @ F.normalize(self.W, dim=1).T                          # (batch, n_designs)
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        target = torch.cos(theta.gather(1, y[:, None]) + MARGIN)          # true design: angle + margin
        return SCALE * cos.scatter(1, y[:, None], target)


@torch.no_grad()
def val_recall(model, dl, codes, device, k=10) -> float:
    """Embed all val photos, search among val, Recall@k on siblings (same design)."""
    model.eval()
    E = []
    for x, _ in dl:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            E.append(model(x.to(device)).float())
    E = torch.cat(E)
    codes = torch.tensor(codes, device=device)
    hits = []
    for i in range(0, len(E), 1024):
        sims = E[i:i + 1024] @ E.T
        rows = torch.arange(sims.shape[0], device=device)
        sims[rows, rows + i] = -2                                          # remove the query itself
        top = sims.topk(k, dim=1).indices
        q = codes[i:i + 1024]
        has_sibling = (codes[None, :] == q[:, None]).sum(1) > 1
        hits.append((codes[top] == q[:, None]).any(1)[has_sibling])
    return torch.cat(hits).float().mean().item()


def main() -> None:
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pl.read_parquet(SPLITS)
    train = df.filter(pl.col("split") == "train")
    code_idx = {c: i for i, c in enumerate(sorted(train["product_code"].unique().to_list()))}
    train = train.with_columns(pl.col("product_code").replace_strict(code_idx).alias("label"))
    val = df.filter(pl.col("split") == "val").with_columns(pl.lit(0).alias("label"))
    print(f"{len(train):,} train photos of {len(code_idx):,} designs | {len(val):,} val photos")

    train_dl = DataLoader(Designs(train, TRAIN_TF), batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                          pin_memory=True, persistent_workers=True, drop_last=True)
    val_dl = DataLoader(Designs(val, get_transforms(train=False)), batch_size=128, shuffle=False,
                        num_workers=2, pin_memory=True)
    val_codes = val["product_code"].to_list()

    model, arc = Embedder().to(device), ArcFace(len(code_idx)).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": list(model.head.parameters()) + list(arc.parameters()), "lr": LR_HEADS},
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    loss_fn = nn.CrossEntropyLoss()

    print(f"before training: val Recall@10 {val_recall(model, val_dl, val_codes, device):.3f}")
    history, best, bad = [], -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}"):
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                emb = model(x)
            loss = loss_fn(arc(emb.float(), y), y)          # ArcFace maths in float32 (acos is touchy in fp16)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(x)
            n += len(x)

        recall = val_recall(model, val_dl, val_codes, device)
        history.append({"epoch": epoch, "train_loss": running / n, "val_recall@10": recall,
                        "minutes": round((time.time() - t0) / 60, 1)})
        (RUNS / "metric_history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch {epoch} | train loss {running / n:.3f} | val Recall@10 {recall:.3f}")

        if recall > best:
            best, bad = recall, 0
            torch.save(model.state_dict(), RUNS / "metric_arcface.pt")
            print("  new best: weights saved")
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"No improvement for {PATIENCE} epochs, stopping.")
                break


if __name__ == "__main__":
    main()