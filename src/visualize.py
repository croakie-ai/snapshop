"""Two explainability figures for the README / report.

gradcam.png       WHERE the tagger looks when it picks the product group and the colour (Grad-CAM)
tsne_designs.png  ImageNet vs ArcFace fingerprints of 25 designs x their colour variants (t-SNE):
                  does metric learning really pull the colours of one design together?

Run from the project root:  python src/visualize.py      (~2-3 minutes; GPU optional)
"""
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.manifold import TSNE

from data.images import IMG_DIR, SPLITS, class_lists
from predict import PROC, ROOT, load_models, prepare

FIG = ROOT / "reports" / "figures"
CAM_GROUPS = ["Garment Upper body", "Garment Lower body", "Garment Full body", "Shoes"]   # 2 val photos each
SEED = 7


# ---------------------------------------------------------------- Grad-CAM
def gradcam(tagger, x, head: str):
    """Grad-CAM for the predicted class of one head -> (7 x 7 heat map in [0, 1], class index).

    layer4 = the last convolutional block: 2,048 feature maps of 7 x 7, the most "semantic" ones.
    weight of each map = average gradient of the class score w.r.t. that map (how much it matters);
    heat map = ReLU(weighted sum of the maps) (keep only evidence FOR the class)."""
    store = {}

    def hook(_, __, out):
        store["maps"] = out
        out.register_hook(lambda grad: store.__setitem__("grads", grad))

    handle = tagger.backbone.layer4.register_forward_hook(hook)
    feats = tagger.backbone(x)
    logits = (tagger.group_head if head == "group" else tagger.colour_head)(feats)
    cls = int(logits.argmax(1))
    tagger.zero_grad()
    logits[0, cls].backward()                                         # gradients of this one class score
    handle.remove()

    weights = store["grads"].mean(dim=(2, 3), keepdim=True)          # (1, 2048, 1, 1)
    cam = F.relu((weights * store["maps"]).sum(dim=1))[0]             # (7, 7)
    return (cam / (cam.max() + 1e-8)).detach().cpu(), cls


def gradcam_figure(tagger, aids: list, device: str, out_path):
    groups, colours = class_lists()
    fig, axes = plt.subplots(3, len(aids), figsize=(2.2 * len(aids), 7.2), squeeze=False)
    for col, aid in enumerate(aids):
        img = Image.open(IMG_DIR / aid[:3] / f"{aid}.jpg").convert("RGB")
        shown = img.resize((256, 256)).crop((16, 16, 240, 240))       # what the model sees (centre 224)
        x = prepare(img).to(device)
        axes[0, col].imshow(shown)
        for row, head, names in [(1, "group", groups), (2, "colour", colours)]:
            cam, cls = gradcam(tagger, x, head)
            cam = F.interpolate(cam[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
            axes[row, col].imshow(shown)
            axes[row, col].imshow(cam.numpy(), cmap="jet", alpha=0.45)
            axes[row, col].set_title(names[cls].replace("Garment ", ""), fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    axes[1, 0].text(-20, 112, "group", rotation=90, va="center", ha="right", fontsize=10, weight="bold")
    axes[2, 0].text(-20, 112, "colour", rotation=90, va="center", ha="right", fontsize=10, weight="bold")
    fig.suptitle("Grad-CAM: where the tagger looks (red = most important for the predicted tag)", weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def pick_cam_photos(n_per_group=2) -> list:
    df = pl.read_parquet(SPLITS).filter(pl.col("split") == "val")
    aids = []
    for g in CAM_GROUPS:
        ids = df.filter(pl.col("group") == g)["article_id"].sample(n_per_group, seed=SEED).to_list()
        aids += [a for a in ids if (IMG_DIR / a[:3] / f"{a}.jpg").exists()]
    return aids


# ---------------------------------------------------------------- t-SNE
def tsne_2d(X: np.ndarray) -> np.ndarray:
    return TSNE(n_components=2, perplexity=min(30, len(X) // 4), init="pca", random_state=0).fit_transform(X)


def tsne_figure(panels: list, codes: list, out_path):
    """panels: [(title, embeddings (n, d))]; codes: design id of each row (same colour = same design)."""
    uniq = sorted(set(codes))
    palette = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors
    colour_of = {c: palette[i % len(palette)] for i, c in enumerate(uniq)}
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 6))
    for ax, (title, X) in zip(np.atleast_1d(axes), panels):
        P = tsne_2d(X)
        ax.scatter(P[:, 0], P[:, 1], c=[colour_of[c] for c in codes], s=45, edgecolors="white", linewidths=0.5)
        ax.set_title(title, weight="bold")
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"t-SNE of {len(codes)} val products from {len(uniq)} designs "
                 "(one colour = one design in its different colourways)", weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def pick_designs(meta: dict, n_designs=25, min_variants=4, seed=0):
    """Val designs with at least min_variants colourways -> row indices + their design ids."""
    df = pl.DataFrame({"row": range(len(meta["ids"])), "code": meta["product_code"], "split": meta["split"]})
    counts = (df.filter(pl.col("split") == "val").group_by("code").len()
                .filter(pl.col("len") >= min_variants).sort("code"))
    chosen = counts["code"].sample(min(n_designs, len(counts)), seed=seed).to_list()
    rows = df.filter(pl.col("code").is_in(chosen) & (pl.col("split") == "val")).sort("code")
    return rows["row"].to_list(), rows["code"].to_list()


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tagger, _ = load_models(device)
    out = FIG / "gradcam.png"
    gradcam_figure(tagger, pick_cam_photos(), device, out)
    print(f"saved {out}")

    imagenet, metric = torch.load(PROC / "emb_imagenet.pt"), torch.load(PROC / "emb_metric.pt")
    assert imagenet["ids"] == metric["ids"]                           # same product order in both files
    rows, codes = pick_designs(metric)
    out = FIG / "tsne_designs.png"
    tsne_figure([("ImageNet ResNet-50 (baseline)", imagenet["emb"][rows].float().numpy()),
                 ("ArcFace (ours)", metric["emb"][rows].float().numpy())], codes, out)
    print(f"saved {out}")
