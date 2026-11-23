"""Tag ANY clothing photo with product group + colour (SnapShop output 2, Phase 3).

The photo goes through exactly the same steps as the training images:
white-pad to square -> resize 256 -> centre-crop 224 -> ImageNet normalize.

Run from the project root:
    python src/predict.py path/to/photo.jpg
    python src/predict.py path/to/photo.jpg --weights runs/finetuned_resnet50_ls.pt --top 3
"""
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps

from data.images import class_lists, get_transforms
from data.make_images import SIZE, pad_to_square
from train_finetune import MultiHeadResNet

ROOT = Path(__file__).resolve().parents[1]


def load_model(weights: Path, device: str) -> MultiHeadResNet:
    model = MultiHeadResNet()
    model.load_state_dict(torch.load(weights, map_location=device))
    return model.to(device).eval()


def prepare(path: Path) -> torch.Tensor:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")      # phone photos: apply the rotation stored in the file
    img = pad_to_square(img).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    return get_transforms(train=False)(img).unsqueeze(0)    # (1, 3, 224, 224): a batch of one


@torch.no_grad()
def predict(model, x, device, top):
    out_g, out_c = model(x.to(device))
    groups, colours = class_lists()
    result = {}
    for task, out, names in [("group", out_g, groups), ("colour", out_c, colours)]:
        p = out.softmax(dim=1)[0].cpu()
        conf, idx = p.topk(top)
        result[task] = [(names[i], float(c)) for c, i in zip(conf, idx)]
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="SnapShop tagger: product group + colour for any photo")
    ap.add_argument("image", type=Path)
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "finetuned_resnet50.pt")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    result = predict(model, prepare(args.image), device, args.top)

    print(f"\n{args.image.name}")
    for task, preds in result.items():
        print(f"  {task}:")
        for name, c in preds:
            print(f"    {name:<22} {c * 100:5.1f}%")


if __name__ == "__main__":
    main()