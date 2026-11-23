# SnapShop — Decision Log

Format: Decision · Options considered · Evidence · Reason · Impact

---

## D1. Implementation language: Python (not C/C++/Java)
- **Options:** C++ (LibTorch) as the compendium specified; Python
- **Evidence:** Faculty approval obtained (Oct 2026)
- **Reason:** Mature tooling (PyTorch, torchvision, FAISS); lets the effort go into experiments, not build systems
- **Impact:** All models in PyTorch (see D15)

## D2. Catalogue split grouped by product_code, not article_id
- **Options:** random split by article_id; grouped split by product_code
- **Evidence:** 105,542 unique article_id but only 47,224 product_code (~2.2 colour variants per design). Example: 0108775015 / 044 / 051 are the same "Strap top" in black, white and off-white
- **Reason:** A random split puts near-identical variants in both train and test → data leakage → inflated scores
- **Impact:** All variants of a design stay on the same side of the split (applied in Phase 1, Step 6)

## D3. Classification targets: product group + colour; pattern postponed; index group excluded
- **Options:** product_group (19), product_type (131), colour_group (50), perceived_colour_master (20), graphical_appearance (30), index_group (5)
- **Evidence:** product_type has 131 classes, many tiny; graphical_appearance includes labels not visible at 224px (Slub, Neps, Treatment, "Other pattern"); index_group is a business section (Ladieswear vs Divided)
- **Reason:** Keep targets that are visually observable and have enough examples per class (~500+ → ~75 test images per class)
- **Impact:** Two targets for Phase 4; pattern revisited later as an optional third target

## D4. Product group: keep 9 classes, drop 10 rare/out-of-scope groups
- **Options:** keep all; merge rare into "Other"; drop; semantic merge
- **Evidence:** 10 smallest groups total 298 products (0.3%): Unknown, Cosmetic, Bags, Items, Furniture, Garment and Shoe care, Stationery, Interior textile, Fun, Underwear/nightwear
- **Reason:** Junk or non-fashion items; an "Other" bucket would be visually incoherent and act as a dumping ground for uncertain predictions
- **Impact:** 9 classes, from Garment Upper body (42,741) down to Nightwear (1,899)

## D5. "Underwear/nightwear" (54 items): dropped after a label audit
- **Options:** merge into Underwear (initial guess); merge into Nightwear; drop
- **Evidence:** Text audit: 48 "Sleeping sack" + 6 "Sleep Bag". Visual audit: all padded sleeveless baby sleeping bags, visually unlike Nightwear (pyjama sets, nightshirts). Figures: reports/figures/label_audit/underwear_nightwear_54.png, nightwear_real.png
- **Reason:** The initial merge guess was wrong; no class fits visually
- **Impact:** 54 products removed. Lesson: audit labels before merging, since names are not evidence

## D6. Colour: use perceived_colour_master_name; drop unknowns; merge 2 tiny classes
- **Options:** colour_group_name (50 shades); perceived_colour_master_name (20 base colours)
- **Evidence:** 16 base colours have 1,100+ products each; "Unknown" 685 + "undefined" 105 have no colour; Yellowish Green 5, Bluish Green 3
- **Reason:** Fewer, larger, more photo-robust classes (lighting blurs fine shades)
- **Impact:** 16 colour classes; Yellowish Green → Green, Bluish Green → Turquoise; 790 products dropped

## D7. Keep "Metal" as a colour class, with a known risk
- **Evidence:** Metal = Gold 1,377 / Silver 709 / Bronze-Copper 94. 85% are Accessories (1,858), 230 Shoes, ~86 garments. Visual audit: clear on earrings, rings and a silver boot; near-invisible on thin necklaces; sunglasses labelled Metal (frame) while lenses dominate visually (green/brown). Figure: reports/figures/label_audit/colour_metal.png
- **Reason:** A real visible property for most items
- **Risks:** shortcut learning (jewellery → "Metal"); label noise (material ≠ dominant visible colour)
- **Impact:** Phase 4 reports Metal accuracy separately for garments vs jewellery vs eyewear

## D8. Drop whole products with invalid labels (not masked loss)
- **Options:** drop the row; keep the row and mask the missing label in the loss
- **Evidence:** Total dropped 1,088 of 105,542 (1.0%); 104,454 retained
- **Reason:** Simplicity; the data loss is negligible
- **Impact:** Output saved as data/processed/labels_clean.parquet (article_id, product_code, group, colour)

## D9. Recommender split by time, not at random
- **Options:** random split of purchases; time split (train on the past, test on a later week)
- **Evidence:** transactions run Sep 2018 – Sep 2020; the last week (16–22 Sep 2020) is held out as the test week
- **Reason:** a recommender predicts the FUTURE; a random split leaks tomorrow's purchases into training
- **Impact:** Phase 6 trains on purchases before 16 Sep 2020 and scores MAP@12 / NDCG@12 on the test week

## D10. Recommender is photo-driven (session-based)
- **Options:** personalised only (needs customer history); photo-driven sequence input
- **Evidence:** SnapShop users may have no H&M purchase history; the input is a photo
- **Reason:** feeding the top search match as the input sequence links the recommender to the photo and works with zero history
- **Impact:** same model and training; evaluation adds a "1-item history" test
## D11. Co-purchase counting added as a recommender baseline
- **Reason:** with 1-item (photo) input, popularity ignores the photo entirely; "most often bought after this item" is the fair non-neural opponent
- **Impact:** Phase 6 compares popularity → co-purchase → LSTM → transformer, on 1-item and long-history customers

## D12. 440 products without images excluded from vision tasks
- **Evidence:** zip contains 105,100 images; 440 of the 104,454 cleaned products have none
- **Reason:** nothing to train on or search with
- **Impact:** 104,014 products for the classifier and search; purchase history left untouched

## D13. Resize by white-padding to square, then 256×256
- **Evidence:** 200-image sample: height ~1,750 px; width 667–2,805 px (most common 1166×1750); some images are wider than tall
- **Options:** squash (distorts), center crop (cuts wide multi-pack shots), pad + resize
- **Reason:** padding keeps the whole product undistorted; white matches H&M's background
- **Impact:** cached at 256×256 so training can random-crop 224×224 (augmentation); originals kept in the zip, so the cache can be regenerated

## D14. Catalogue split: stratified group split, 7 folds → ~71/14/14
- **Options:** random split by article_id (leaks); group split (risks uneven classes); stratified group split
- **Reason:** groups by product_code (no leakage) + stratifies by product group (every class fairly represented)
- **Evidence:** script verifies 0 designs span more than one split; per-class val/test shares ≈14%
- **Impact:** data/processed/splits.parquet, seed 42, reproducible

## D15. Phase 2 MLP in PyTorch, not from-scratch NumPy
- **Options:** NumPy forward/backward by hand; PyTorch (autograd)
- **Reason:** student's choice (22 Nov 2026); viva "modify" questions are expected to be conceptual (change activation / layers / optimizer)
- **Risk accepted:** the compendium asks for from-scratch fundamentals; covered instead by a hand-worked forward/backward example (softmax + cross-entropy, gradient p − y) in the maths sheet
- **Impact:** notebooks/04_mlp.ipynb; runs/mlp_group_best.pt, runs/mlp_colour_best.pt

## D16. DeepFashion2 source: Kaggle copy (yashwantk23cse/deep-fashion)
- **Why:** official access form submitted 21 Nov 2026, no password received; others report no replies
- **Verified:** 15.8 GB; official folder layout; sample label file has boxes, masks, categories, source (user/shop) and pair_id
- **Use:** detector trained on a ~30k subset of DF2 train; DF2 validation kept for evaluation only (no leakage)
- **Licence note:** non-commercial academic use; the official DeepFashion2 paper is cited; no images redistributed

## D17. Standardize MLP inputs (train-only mean/std)
- **Evidence:** raw 0–1 inputs: bumpy loss, best val acc 78.7% at epoch 58; standardized: smooth, 83.4% at epoch 10
- **Reason:** inputs were all positive and mostly near 1 (white background) → zig-zag updates
- **Impact:** MLP baseline (product group): val acc 83.4%, macro-F1 0.730; Nightwear weakest (F1 0.393, recall 0.289), as predicted by the label audit

## D18. CNN step 1: frozen ResNet-50 features + linear heads (baseline for fine-tuning)
- **Options:** train a CNN from scratch; frozen pretrained ResNet-50; fine-tuned ResNet-50
- **Method:** ImageNet ResNet-50 with fc removed → 2,048-number fingerprint per image, extracted once (src/data/extract_features.py → resnet50_features.pt); standardized with train mean/std; one linear layer per task (SGD + momentum, early stopping on val loss)
- **Evidence (val):** group 89.3% acc / macro-F1 0.835 (MLP 83.4% / 0.730); colour 70.6% / 0.575 (MLP 71.6% / 0.599); Metal F1 rose to 0.736
- **Reason:** cheap test of "how far do ImageNet features go without changing them"
- **Finding:** frozen features help shape (group) but not colour: ImageNet filters learned to ignore colour shifts. Metal gain is consistent with the shortcut risk in D7 (it can spot jewellery shapes)
- **Impact:** runs/frozen_group_head.pt, runs/frozen_colour_head.pt; motivates fine-tuning (D19)

## D19. CNN step 2: fine-tune ResNet-50, one backbone with two heads (multi-task)
- **Options:** two separate fine-tuned networks; one shared backbone + group head + colour head
- **Method:** loss = group CE + colour CE; AdamW, LR 1e-4 for the backbone (gentle, avoids catastrophic forgetting) and 1e-3 for the new heads; mixed precision (fp16); batch 64; max 10 epochs, early stopping (patience 3) on val loss
- **Augmentation:** RandomCrop 224 from 256 + horizontal flip only. NO colour jitter, because colour is a label
- **Reason:** half the training time and one forward pass at demo time; shared features can help both tasks
- **Evidence (val, best = epoch 4 of 7):** group 93.7% / macro-F1 0.901; colour 82.0% / 0.741 (vs frozen 89.3% / 0.835 and 70.6% / 0.575). After epoch 4 train loss kept falling (0.55 → 0.32) while val loss rose (0.79 → 0.94): overfitting; early stopping ended the run at epoch 7 and kept epoch 4
- **Impact:** src/train_finetune.py → runs/finetuned_resnet50.pt (epoch 4) + runs/finetune_history.json; fine-tuning gave the biggest gain on colour (+11.4 points over frozen)

## D20. Validation DataLoader: 2 workers, not persistent
- **Evidence:** run 1 crashed at the first validation with Windows error 1455 (out of memory): 4 persistent train workers + 4 val workers + a leftover notebook kernel on 16 GB RAM
- **Reason:** each worker is a separate Python process holding its own copy of PyTorch
- **Impact:** loading speed only; the same images, order and transforms, so results are unchanged

## D21. Label smoothing (0.1) for the classifier: run B is the current best
- **Why tried:** the error gallery showed the most confident mistakes are mostly wrong H&M labels (e.g. light blue jeans labelled Black); hard 100% targets make the model memorise that noise
- **Method:** same as D19, only change: CrossEntropy(label_smoothing=0.1) for training; plain CrossEntropy for val so runs compare fairly; train workers 4 -> 2 (RAM, error 1455)
- **Evidence (val):** run B best = epoch 6: group 94.1% / macro-F1 0.910, colour 82.3% / 0.752 (run A: 93.7% / 0.901, 82.0% / 0.741). Overfitting started 2 epochs later than run A (best epoch 4 -> 6)
- **Note:** val loss is not comparable between A and B (smoothing lowers confidence on correct answers too); compared on accuracy + macro-F1
- **Impact:** runs/finetuned_resnet50_ls.pt used by predict.py; final classifier is locked only after the weighted-loss run, then the test set is used once

## D22. Detector: YOLO11s-seg on a 30k DeepFashion2 subset
- **Options:** Faster/Mask R-CNN; YOLO-seg (boxes + masks in one fast model)
- **Data:** 30,000 random DF2 train images; 2,000 random DF2 validation images for epoch selection; the other ~30k DF2 validation images kept untouched for final evaluation (ids saved: train_ids.txt, val_ids.txt; seed 42). Converted by src/data/make_yolo_df2.py (multi-part outlines joined with ultralytics' merge_multi_segment)
- **Training:** COCO-pretrained yolo11s-seg, imgsz 640, batch 8, workers 2, planned 25 epochs (patience 6); resumed once after a laptop restart; stopped by choice after epoch 23 (gain per epoch had fallen to ~0.002-0.004 mAP)
- **Evidence (2k val, epoch 23):** box mAP50 0.786 / mAP50-95 0.696; mask mAP50 0.616 / mAP50-95 0.407. For reference, the DeepFashion2 paper's Mask R-CNN (391k images, full val): box AP 0.667, mask AP 0.680
- **Finding:** boxes are at benchmark level; masks lag. Suspects: the polygon-merging step in our converter, YOLO's lower mask resolution, less data. To check in the Phase 5 evaluation; the crop ablation (full vs box vs mask) decides whether masks matter for search
- **Impact:** runs/segment/runs/yolo/df2_y11s/weights/best.pt

## D23. Visual search: hybrid of ImageNet and fine-tuned fingerprints (alpha = 0.5)
- **Method:** every product (104,014) -> 2,048-d backbone features, L2-normalised (src/extract_embeddings.py); exact cosine search with FAISS IndexFlatIP over the whole catalogue
- **Test:** 11,437 val queries that have a sibling (same product_code, other colour); Recall@K = a sibling in the top K; plus group / colour match in the top 10; plus galleries of 5 fixed queries (seed 42)
- **Evidence:** Recall@10 / colour match@10: ImageNet 0.360 / 0.461; fine-tuned A 0.181 / 0.764; fine-tuned B 0.159 / 0.782. Fine-tuning for tags organised the features by colour and type and lost fine design detail (label smoothing slightly more so)
- **Key insight:** sibling recall and colour match pull in opposite directions by definition (a sibling is always another colour), so no single metric decides
- **Hybrid:** score = alpha * cos(ImageNet) + (1 - alpha) * cos(fine-tuned B), done by concatenating sqrt-weighted fingerprints. alpha 0 / 0.25 / 0.5 / 0.75 / 1 -> Recall@10 0.159 / 0.185 / 0.215 / 0.261 / 0.360, colour match 0.782 / 0.775 / 0.759 / 0.718 / 0.461
- **Choice:** alpha = 0.5: keeps colour (0.759) and type (0.927) match near the maximum while design recall rises 35%; galleries look right on all 5 queries (same type, same colour, siblings near the top). Re-checked after metric learning
- **Impact:** notebooks/07_search.ipynb; figures search_*.png
