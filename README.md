# Conjunctival anaemia screening — a comparison study

Smartphone-style photos of the inner lower eyelid (palpebral conjunctiva) → anaemic / not anaemic / uncertain,
against the WHO 2024 age- and sex-specific Hb cutoffs. The project compares colour-feature and deep-learning
methods, preprocessing choices and evaluation methods on CP-AnemiC (Ghana, children) and Eyes-Defy-Anemia
(India and Italy, adults), and ships a demo app.

## Folder layout

| Folder | Contents |
|---|---|
| `data/raw/` | The two datasets as downloaded and extracted (not shared; see their licences) |
| `data/processed/` | `labels.csv`, `features.csv` from the first baseline (10 Nov) |
| `kaggle/anemia-data/` | Standardised copy of both datasets: `labels.csv`, `conj/`, `eye/`, `mask/`, `groups.csv`. **All study scripts read from here.** |
| `kaggle/anemia-data.zip` | The same, zipped, for upload to Kaggle as a private dataset |
| `src/near_duplicates.py` | Finds near-duplicate images in CP-AnemiC (same eye mirrored or re-cropped) |
| `src/study/` | The comparison study: features, shared test code, experiments A–H, report |
| `src/screening/` | What the demo runs: quality check, pipeline, final model training |
| `src/segmentation/` | CPU segmentation model (works without a GPU) |
| `src/perfusion/` | Analysis for the perfusion (blood-flow) experiment; recording steps in `data/perfusion/README.md` |
| `notebooks/` | Kaggle notebooks: 01 (first model comparison), 02 (deep methods), 03 (segmentation) |
| `models/` | Trained models used by the demo (`screening_models.joblib`, `conj_seg.onnx`) |
| `results/study/` | Every table and figure of the study, plus `STUDY_RESULTS.md` |
| `app/app.py` | The demo app |

## How to run everything (laptop, CPU)

```
pip install -r requirements.txt
python src/near_duplicates.py          # 5-10 min   -> results/study/groups.csv
python src/study/features_v2.py        # 5-10 min   -> results/study/features_v2.csv
python src/study/exp_protocol.py       # A: evaluation methods
python src/study/exp_models.py         # B: 8 models x 6 feature sets (+ deep features if present)
python src/study/exp_preprocessing.py  # C: colour correction
python src/study/exp_abstention.py     # E: "uncertain" answers
python src/study/exp_hb.py             # F: Hb number
python src/study/exp_quality.py        # G: quality check
python src/study/exp_segmentation.py   # H: automatic vs hand-drawn masks (needs notebook 03 output)
python src/study/report.py             # tables, figures, results/study/STUDY_RESULTS.md
python src/screening/train_final.py    # models used by the app
python src/segmentation/train_seg_cpu.py   # eyelid segmentation, about 40 min on CPU -> models/conj_seg.onnx
python src/perfusion/perfusion_analysis.py # once the volunteer photos are in data/perfusion/
python app/app.py                      # demo at http://127.0.0.1:7860  (add --lan to open it from a phone)
```
Experiments A–G take about 1 hour in total on a laptop. Everything is already run; the outputs are in `results/study/`.

## The tests every method is scored on

| Test | Train | Test | Why |
|---|---|---|---|
| Ghana (copies grouped) | 4/5 of the 392 Ghana images | the other 1/5, rotated, 3 repeats | near-duplicates of one eye are never on both sides |
| India | 4/5 of India | the other 1/5 | India and Italy are always scored **separately** |
| Italy | 4/5 of Italy | the other 1/5 | |
| Ghana → India, Ghana → Italy | all Ghana | India / Italy | new population, new camera |
| India+Italy → Ghana | India + Italy | Ghana | the reverse |

"Fair average" = mean AUC of these six. "Cross-population average" = mean of the last three.
95% confidence intervals and paired comparisons use 1,000 bootstrap resamples (copies of one eye resampled together).

## Main findings so far (18 Nov 2026) — details in `results/study/STUDY_RESULTS.md`

1. **The evaluation method changes the answer more than the model does.** Random forest on Ghana scores AUC
   0.91 with the random split most CP-AnemiC papers use, 0.80 after removing exact copies, 0.76 with near-copies
   kept together, and 0.61 when trained on India + Italy. Pooling India and Italy gives 0.88; India alone is 0.66.
2. **No single best model.** The top methods all sit at a fair average of about 0.74. Tree models (random forest,
   extra trees, LightGBM) with texture features are significantly better *within* Ghana (+0.10 to +0.13 AUC,
   p < 0.01) but significantly worse when moved to a new population (Ghana → Italy −0.11 to −0.17).
   Logistic regression on brightness-free colour ratios is the best across populations (0.74).
3. **Software colour correction did not help.** Grey-world, white-patch and a sclera (white of the eye)
   reference all scored below no correction on Eyes-Defy, most clearly when moving between India and Italy.
4. **Allowing "uncertain" raises accuracy** within a site (Ghana 73% → 87% on the 35% of cases answered), but the
   uncertainty thresholds do not transfer to a new population (Ghana → India stays below 50%).
5. **The Hb number is only modestly predictable:** average error 1.2–1.5 g/dL within a site vs 1.6–1.7 for
   guessing the average. Across sites: 2.1 vs 2.5 g/dL from Ghana to India+Italy, no better than guessing the
   other way round.
6. **Quality check:** catches 100% of blurred, too-dark, glare and barely-visible-eyelid images and 86% of motion
   blur, while rejecting 0% of real Eyes-Defy photos (8% of the softer, low-resolution Ghana cut-outs).

7. **Automatic segmentation works.** A U-Net with a pretrained encoder (notebook 03, Kaggle GPU) finds the eyelid
   with Dice 0.85 (India 0.84, Italy 0.86) in 5-fold cross-validation over all 217 Eyes-Defy photos, none below 0.5.
   Re-running the anaemia tests on these automatic masks gives the same accuracy as the hand-drawn masks
   (average AUC 0.78 vs 0.78). The app uses this model (`models/conj_seg.onnx`); the earlier CPU model is kept as
   `models/conj_seg_cpu.onnx`. Caveat: Eyes-Defy photos were taken in a controlled set-up, so phone photos taken at
   home may segment less well; the app lets you paint the region by hand when that happens.

8. **Deep learning does not beat colour features.** Four pretrained networks used as feature extractors
   (ResNet-50, EfficientNet-B0, ConvNeXt-Tiny, DINOv2-small, each with the same 8 models) and two fine-tuned
   networks (EfficientNet-B0, ConvNeXt-Tiny) were scored on the same fair tests (notebook 02, Kaggle GPU).
   Best of each family, fair average AUC: colour/texture features 0.741, DINOv2 features + LightGBM 0.740,
   fine-tuned ConvNeXt-Tiny 0.734. Across populations the simple colour-ratio logistic regression stays best (0.738,
   vs 0.726 for the fine-tuned ConvNeXt and 0.697 for DINOv2). With about 600 training images, the networks
   add complexity without adding accuracy.


## The demo app

`python app/app.py` → two tabs:
- **Screen a photo:** upload or take a photo of the pulled-down lower eyelid, enter age and sex. The eyelid region
  is found automatically (`models/conj_seg.onnx`) or painted with the brush. The quality check runs first; then
  the main method (logistic regression · ratio colour) gives anaemic / not anaemic / uncertain, and the other top
  methods are shown side by side.
- **Study results:** every figure and table of the study.

## Earlier baseline (10 Nov 2026)

Kept for reference in `results/baseline_*.csv`. Superseded by the study above.

## Data issues found

1. **CP-AnemiC contains duplicated images.** 318 of the 710 images share their picture
   with at least one other record (87 groups of pixel-identical files, plus a few re-saved
   copies), and the copies carry different Hb, age, sex and hospital. Their labels cannot
   be trusted, so the pipeline uses only the 392 images that appear once (`usable = 1`).
   With the duplicates left in, a random forest scores AUC 0.90 by memorising copies.
2. **Eyes-Defy pooled results are inflated by site.** India is mostly anaemic (68 of 95)
   and Italy mostly not (23 of 122), and the two sites differ in colour, so a pooled model
   partly learns the site. Always report India and Italy separately.
3. **Eyes-Defy cut-outs come in three formats** (transparent background, white background,
   one 16-bit file) and four have misspelt file names. `features.py` handles all of them.
4. **Italy subject 93 has no Hb value** and is excluded (217 of 218 subjects used).
5. **WHO 2024 cutoffs change 17 CP-AnemiC labels** relative to the dataset's own
   Anemic / Non-anemic folders (the cutoff for 6–23 months is now 10.5 g/dL).
6. **CP-AnemiC images are already cropped to the conjunctiva**, so sclera-based
   normalisation can only be tested on Eyes-Defy.

7. **CP-AnemiC also contains near-duplicates** (found 18 Nov): the same eye mirrored or re-cropped, filed under
   different children with different Hb. Among the 392 images left after removing exact copies, 26 (strict
   matching) to 71 (lenient) share an eye with another image. `src/near_duplicates.py` finds them with SIFT keypoint
   matching; `results/study/figures/fig6_duplicates.jpg` shows examples.
