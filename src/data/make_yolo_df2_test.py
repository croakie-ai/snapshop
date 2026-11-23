"""YOLO test split (Phase 5.3b): 5,000 DeepFashion2 validation images YOLO has NEVER used.

YOLO trained on 30k DF2 *train* images and picked its best epoch with 2,000 DF2 *validation* images
(val_ids.txt). Those 2,000 make its scores slightly optimistic, so here we take 5,000 OTHER validation
images (seed 42), convert them the same way, and add them to df2.yaml as the "test" split.

Run from the project root:  python src/data/make_yolo_df2_test.py
Then:  yolo segment val model=runs/segment/runs/yolo/df2_y11s/weights/best.pt
         data=data/processed/df2_yolo/df2.yaml split=test imgsz=640 batch=8 workers=2
         project=runs/yolo name=df2_test exist_ok=True
"""
import json
import random
import shutil

from PIL import Image
from tqdm import tqdm

from make_yolo_df2 import DF2, OUT, SEED, to_yolo_lines

N_TEST = 5_000


def main() -> None:
    used = set((OUT / "val_ids.txt").read_text().split())                 # the 2,000 used for epoch picking
    ids = sorted(p.stem for p in (DF2 / "validation" / "image").glob("*.jpg"))
    chosen = random.Random(SEED).sample([i for i in ids if i not in used], N_TEST)

    (OUT / "images" / "test").mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / "test").mkdir(parents=True, exist_ok=True)
    for i in tqdm(chosen, desc="test"):
        img_src = DF2 / "validation" / "image" / f"{i}.jpg"
        anno = json.loads((DF2 / "validation" / "annos" / f"{i}.json").read_text())
        w, h = Image.open(img_src).size
        shutil.copy2(img_src, OUT / "images" / "test" / f"{i}.jpg")
        (OUT / "labels" / "test" / f"{i}.txt").write_text("\n".join(to_yolo_lines(anno, w, h)))
    (OUT / "test_ids.txt").write_text("\n".join(chosen))

    yaml = (OUT / "df2.yaml").read_text().splitlines()
    if not any(line.startswith("test:") for line in yaml):
        yaml.insert(yaml.index("val: images/val") + 1, "test: images/test")
        (OUT / "df2.yaml").write_text("\n".join(yaml))
    print(f"test: {N_TEST:,} untouched DF2 validation images (none of the 2,000 used to pick YOLO's epoch)")


if __name__ == "__main__":
    main()
