"""Fine-tune ResNet-50 with TWO heads (product group + colour) on the product photos (Phase 3).

- One shared ResNet-50 backbone (ImageNet-pretrained), all layers learning.
- Two heads on its 2,048-number output: group (9 classes) and colour (16 classes).
- Loss = group loss + colour loss, so one pass of ResNet answers both (multi-task learning).
- Small learning rate for the pretrained backbone (adapt gently, don't forget ImageNet),
  bigger one for the new heads (they start from random).
- Mixed precision (float16) for speed and GPU memory on the RTX 3050.
- Early stopping on validation loss; the best epoch's weights are saved.

Uses : splits.parquet + images_256/ (through data/images.py)
Makes: runs/finetuned_resnet50.pt     best weights (lowest val loss)
       runs/finetune_history.json     per-epoch train loss + val loss/accuracy/macro-F1

Run from the project root:  python src/train_finetune.py
"""
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

from data.images import ProductImages

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

EPOCHS = 10            # maximum; early stopping usually ends it sooner
PATIENCE = 3           # stop after 3 epochs without val improvement
BATCH_SIZE = 64        # lower to 32 if you get "CUDA out of memory"
LR_BACKBONE = 1e-4     # small: gently adapt the pretrained filters
LR_HEADS = 1e-3        # bigger: the heads start from random
SEED = 42
LABEL_SMOOTHING = 0.1   # 0 = original run A; 0.1 = run B
TAG = "_ls"             # added to output filenames so run A's files aren't overwritten


class MultiHeadResNet(nn.Module):
    """ResNet-50 backbone + a group head + a colour head."""

    def __init__(self, n_groups: int = 9, n_colours: int = 16):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()                  # outputs the 2,048-number fingerprint
        self.group_head = nn.Linear(2048, n_groups)
        self.colour_head = nn.Linear(2048, n_colours)

    def forward(self, x):
        f = self.backbone(x)                              # (batch, 2048)
        return self.group_head(f), self.colour_head(f)   # (batch, 9), (batch, 16)


@torch.no_grad()
def evaluate(model, dl, loss_fn, device):
    """Val loss + accuracy and macro-F1 for both tasks."""
    model.eval()
    total, n = 0.0, 0
    g_pred, c_pred, g_true, c_true = [], [], [], []
    for x, g, c in dl:
        x, g, c = x.to(device, non_blocking=True), g.to(device), c.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
            out_g, out_c = model(x)
            loss = loss_fn(out_g, g) + loss_fn(out_c, c)
        total += loss.item() * len(x)
        n += len(x)
        g_pred.append(out_g.argmax(dim=1).cpu()); c_pred.append(out_c.argmax(dim=1).cpu())
        g_true.append(g.cpu()); c_true.append(c.cpu())
    gp, cp, gt, ct = (torch.cat(t).numpy() for t in (g_pred, c_pred, g_true, c_true))
    return {
        "loss": total / n,
        "group_acc": float((gp == gt).mean()), "group_f1": float(f1_score(gt, gp, average="macro")),
        "colour_acc": float((cp == ct).mean()), "colour_f1": float(f1_score(ct, cp, average="macro")),
    }


def main() -> None:
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RUNS.mkdir(exist_ok=True)

    train_dl = DataLoader(ProductImages("train"), batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, persistent_workers=True)   # 2 (not 4): avoids Windows RAM error 1455
    val_dl = DataLoader(ProductImages("val"), batch_size=128, shuffle=False,
                        num_workers=2, pin_memory=True)   # 2 workers, not persistent: avoids Windows RAM error 1455

    model = MultiHeadResNet().to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": list(model.group_head.parameters()) + list(model.colour_head.parameters()), "lr": LR_HEADS},
    ], weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)   # training: "never be 100% sure"
    eval_loss_fn = nn.CrossEntropyLoss()                              # val: plain loss, so runs A and B compare fairly
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    history, best_loss, bad = [], float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for x, g, c in tqdm(train_dl, desc=f"epoch {epoch}"):
            x, g, c = x.to(device, non_blocking=True), g.to(device), c.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                out_g, out_c = model(x)                          # 1. forward
                loss = loss_fn(out_g, g) + loss_fn(out_c, c)     # 2. loss for BOTH tasks
            optimizer.zero_grad(set_to_none=True)                # 3. clear old gradients
            scaler.scale(loss).backward()                        # 4. backward (all 23.5M + heads)
            scaler.step(optimizer)                               # 5. update weights
            scaler.update()
            running += loss.item() * len(x)
            n += len(x)

        val = evaluate(model, val_dl, eval_loss_fn, device)
        record = {"epoch": epoch, "train_loss": running / n,
                  **{f"val_{k}": v for k, v in val.items()},
                  "minutes": round((time.time() - t0) / 60, 1)}
        history.append(record)
        (RUNS / f"finetune{TAG}_history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch {epoch} | train loss {record['train_loss']:.3f} | val loss {val['loss']:.3f} | "
              f"group acc {val['group_acc']:.3f} F1 {val['group_f1']:.3f} | "
              f"colour acc {val['colour_acc']:.3f} F1 {val['colour_f1']:.3f} | {record['minutes']} min")

        if val["loss"] < best_loss:
            best_loss, bad = val["loss"], 0
            torch.save(model.state_dict(), RUNS / f"finetuned_resnet50{TAG}.pt")
            print("  new best val loss: weights saved")
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"No improvement for {PATIENCE} epochs, stopping.")
                break


if __name__ == "__main__":
    main()