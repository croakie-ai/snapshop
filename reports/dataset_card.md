# Dataset Card: SnapShop (H&M catalogue + purchases)

## 1. Source and licence
- **Dataset:** H&M Personalized Fashion Recommendations (Kaggle competition, released 2022)
- **Files used:** `articles.csv` (catalogue), `images/` (product photos), `transactions_train.csv` (purchases); `customers.csv` extracted but not used
- **Licence / terms:** accessed by joining the Kaggle competition and accepting its rules; used for non-commercial academic work only. *(Confirm the exact wording on the competition's Rules tab before submission.)*
- **Privacy:** customer IDs are anonymised hashes; no names or personal details are used.

## 2. Composition
| Part | Size |
|---|---|
| Catalogue products (`article_id`) | 105,542 |
| Distinct designs (`product_code`) | 47,224 (~2.2 colour variants per design) |
| Product photos in the archive | 105,100 (442 products have none) |
| Purchases | 31,788,324 (20 Sep 2018 → 22 Sep 2020, ~105 weeks) |
| Customers with ≥1 purchase | 1,362,281 |
| Products ever bought | 104,547 (995 never bought) |

## 3. Labels (attribute classifier)
- **Targets:** product group (9 classes) and colour (16 classes, from `perceived_colour_master_name`)
- **Cleaning** (decisions D3–D8): dropped 10 junk/out-of-scope groups (298 products, incl. 54 baby sleeping bags found by a label audit) and products with unknown colour (790); merged Yellowish Green → Green, Bluish Green → Turquoise
- **Result:** 104,454 of 105,542 products kept (99.0%); 104,014 of those have an image
- **Class imbalance:** Garment Upper body 42,327 vs Nightwear 1,893 (~22×) → report macro-F1, use a weighted loss

## 4. Splits
- **Catalogue (classifier, search):** stratified group split, seed 42, 7 folds → 1 test / 1 val / 5 train
  - Train 74,295 · Val 14,859 · Test 14,860
  - Grouped by `product_code`: **0** designs span more than one split (verified in code)
  - Stratified by product group: every class ≈ 71.4 / 14.3 / 14.3 (Nightwear: 1,352 / 270 / 271)
- **Purchases (recommender):** time-based. Last week (16–22 Sep 2020) = test, the week before = val, earlier weeks = train. The test week looks like a normal (non-sale) week.

## 5. Preprocessing
- Images padded to square with white (no distortion, nothing cropped), resized to 256×256 (LANCZOS, JPEG quality 90), cached in `data/processed/images_256/`; training random-crops 224×224
- Purchases converted to Parquet (3.49 GB CSV → 470 MB; ~2 s to load)
- All steps are scripts in `src/data/` and can be rerun from the raw files

## 6. Known issues and limitations
- **Label noise:** colour sometimes describes the material rather than the dominant visible colour (e.g. sunglasses labelled "Metal" because of the frame)
- **Shortcut risk:** 85% of "Metal" items are accessories (jewellery), so the model may learn "jewellery → Metal"; tested separately in Phase 3
- **Usage-based classes:** Nightwear is defined by when it's worn, not how it looks (some items are plain T-shirts) → expected confusion with Upper body
- **Fine detail lost at 256 px:** lace, thin chains and small prints; pattern labels postponed partly for this reason
- **Domain gap:** all catalogue photos are clean studio shots; real user photos are not (addressed in Phase 5)
- **Cold start:** 36.8% of customers have ≤5 purchases; 995 products were never bought
- **Seasonality:** sale spikes (late June, late November) and a COVID dip (March 2020) in the purchase history
- **Representation:** a single Swedish retailer's catalogue and customers; styles, sizes and body types may not generalise to other markets (e.g. Indian ethnic wear)

## 7. Intended use
Academic course project (CSAI3027P Deep Learning, UPES). Not for commercial use.
