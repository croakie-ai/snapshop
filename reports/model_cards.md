# SnapShop Model Cards

One card per model in the pipeline. Format follows "Model Cards for Model Reporting" (Mitchell et al., 2019), shortened: what it is, what it was trained on, how well it does, where it fails, and how it should (not) be used.

All numbers are on held-out test data unless marked (val). Every setting was chosen on validation data; the test set was used once (`src/evaluate_test.py`, `runs/test_results.json`). Decisions behind each choice: [`decisions.md`](decisions.md).

---

## 1. Tagger: product group + colour

| | |
|---|---|
| **Task** | One garment photo → product group (9 classes) and colour (16 classes); also gives a 2,048-d colour/type fingerprint used by search |
| **Architecture** | ResNet-50 (ImageNet-pretrained) + two linear heads (2048→9, 2048→16), multi-task |
| **Training data** | H&M product photos: 74,295 train products (white-background catalogue shots), grouped split by `product_code`; labels cleaned and merged from H&M's own (D3–D8) |
| **Training** | AdamW, lr 1e-4 backbone / 1e-3 heads, weight decay 1e-4, batch 64, fp16, label smoothing 0.1, random crop + flip (no colour augmentation: colour is a label), early stopping (best epoch 6) |
| **Weights** | `runs/finetuned_resnet50_ls.pt` (also `runs/onnx/tagger.onnx`, `tagger_int8.onnx`) |
| **Results (test, 14,860 products)** | group **93.9%** accuracy, macro-F1 **0.905**; colour **82.8%**, macro-F1 **0.756** |
| **Baselines (val)** | majority 40.7% / 21.2% · MLP 83.4% / 71.6% · frozen ResNet-50 89.3% / 70.6% |
| **Weakest classes (val F1)** | Nightwear 0.69 (looks like ordinary tops/trousers); colours Mole 0.46, Orange 0.57, Lilac Purple 0.60, Turquoise 0.60 |
| **Known failure modes** | Shortcut learning: jewellery shape → "Metal" (51% of non-metal jewellery, val); worn garments / street photos are out of distribution (fixed in the pipeline by YOLO's garment-type hint, D26); some H&M colour labels are wrong |
| **Intended use** | Tagging catalogue-style or cropped garment photos inside SnapShop |
| **Not for** | Judging people, sizes or body shapes; non-clothing images; commercial use (H&M data licence) |

## 2. Designer: ArcFace design fingerprint

| | |
|---|---|
| **Task** | Garment photo → 512-d L2-normalised fingerprint where colourways of the same design are close |
| **Architecture** | ResNet-50 (ImageNet-pretrained) + Linear(2048→512) + BatchNorm, trained with an ArcFace head (s = 30, m = 0.3) over 33,401 train designs; the ArcFace head is discarded after training |
| **Training** | AdamW, lr 1e-4 / 1e-3, batch 64, 10 epochs (best epoch 9), random crop + flip + brightness/contrast/saturation jitter + 20% grayscale (so colour does not decide the design), early stopping on val Recall@10 |
| **Weights** | `runs/metric_arcface.pt` (also `runs/onnx/designer.onnx`, `designer_int8.onnx`) |
| **Results (test, 11,441 queries vs all 104,014 products)** | Recall@10 **0.801**, Recall@1 0.493, Recall@20 0.854 (a sibling colourway of the same design in the top k) |
| **Baselines (test)** | ImageNet ResNet-50 fingerprint: Recall@10 0.352 · fine-tuned tagger fingerprint (val): 0.181 |
| **Used as** | The design half of the hybrid search: score = 0.5 · cos(design) + 0.5 · cos(colour/type) → same type in top 10: 93%, same colour: 71% |
| **Known failure modes** | Ignores colour by design (hence the hybrid); garments photographed sideways or crumpled search poorly (CNNs are not rotation-invariant); trained on catalogue shots only, so street photos are harder (street-to-shop Recall@10 0.204 with YOLO crops) |

## 3. Detector: YOLO11s-seg on DeepFashion2

| | |
|---|---|
| **Task** | Find each garment in a photo: box + mask + one of 13 DeepFashion2 clothing types |
| **Architecture** | Ultralytics YOLO11s-seg (COCO-pretrained), one-stage, anchor-free, 32 mask prototypes |
| **Training data** | 30,000 DeepFashion2 train images (consumer + shop photos), 2,000 for validation |
| **Training** | Ultralytics defaults (SGD-style optimiser, lr0 0.01, momentum 0.937, weight decay 5e-4, 3 warm-up epochs), batch 8, 640 px, mosaic (off for the last 10 epochs), 23 of 25 epochs (patience 6) |
| **Weights** | `runs/segment/runs/yolo/df2_y11s/weights/best.pt` |
| **Results (5,000 untouched DeepFashion2 photos)** | box mAP50 **0.779**; mask mAP50 0.600 · (val: box mAP50-95 0.696 vs 0.667 for the DeepFashion2 paper's Mask R-CNN) |
| **In the pipeline** | Box crops (+5% margin), confidence ≥ 0.5, at most 3 garments; its garment type becomes the group hint; no detection → the full photo |
| **Known failure modes** | Trousers and shorts masks come out empty (box crops are used instead, D25); no classes for bags, shoes or accessories (falls back to the full photo) |
| **Licence** | Ultralytics YOLO is AGPL-3.0; DeepFashion2 images are research-only and are not redistributed |

## 4. Recommender: LSTM next-item model

| | |
|---|---|
| **Task** | Sequence of purchased items → scores for the 29,882 items in the vocabulary (the next purchase) |
| **Architecture** | Item embedding (64-d, padding index 0) → 1-layer LSTM (64) → LayerNorm → dot product with the same item embeddings (weight tying) |
| **Training data** | H&M transactions up to 8 Sep 2020: 656,182 customer sequences (last 50 items), items bought ≥ 10 times |
| **Training** | Cross-entropy over all items at every position, AdamW lr 1e-3 / weight decay 1e-4, batch 256, fp16, early stopping on val-week MAP@12 (patience 2) |
| **Weights** | `runs/rec_lstm.pt` (alternative: `runs/rec_transformer.pt`, 2 layers, 2 heads, causal mask; ties on val) |
| **Results (test week 16–22 Sep 2020, 68,984 customers)** | MAP@12 **0.0200** (full history), 0.0187 (last item only = the SnapShop case) |
| **Baselines (test)** | co-purchase counts 0.0106 / 0.0090 · last-week popularity 0.0087 |
| **Context** | Absolute MAP@12 is low on H&M (≈2 purchases out of ~30K items a week); the Kaggle winner (large ranking ensemble) reached ≈0.037 on a different week and customer set |
| **Known failure modes** | Items bought < 10 times have no embedding (cold start → this week's best-sellers); repeat purchases and trends not modelled explicitly; only 6 months of history |
| **Not for** | Profiling individual customers; the data is anonymised and only used in aggregate |

## Deployment variants (ONNX)

| Model | fp32 ONNX | INT8 ONNX | Accuracy fp32 → INT8 (1,000 val images) |
|---|---|---|---|
| Tagger | 94.4 MB | 24.5 MB | group 94.7% → 94.3%, colour 82.6% → 81.8% |
| Designer | 98.4 MB | 28.5 MB | fingerprint cosine to fp32: mean 0.975 |

INT8 = static quantization of the convolutions only, per-channel weights, percentile calibration on 100 train images (`src/export_onnx.py`, D29).
